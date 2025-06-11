import os
import time
import math
import json
import argparse
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# Dataset class for algorithmic tasks
class AlgorithmicDataset(Dataset):
    def __init__(self, data, tokenizer, block_size, mask_first_n=0):
        self.data = data
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.mask_first_n = mask_first_n
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Tokenize the equation
        tokens = self.tokenizer.encode(self.data[idx])
        
        # Find the position of '=' to mask loss before it
        eq_pos = -1
        eq_token = self.tokenizer.encode('=')[0]
        for i, token in enumerate(tokens):
            if token == eq_token:
                eq_pos = i
                break
        
        # Pad or truncate to block_size
        if len(tokens) < self.block_size:
            tokens = tokens + [self.tokenizer.pad_token] * (self.block_size - len(tokens))
        else:
            tokens = tokens[:self.block_size]
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1 if t != self.tokenizer.pad_token else 0 for t in tokens]
        
        # Create loss mask
        loss_mask = [0] * len(tokens)
        
        if self.mask_first_n > 0:
            # Mask first N tokens for sanity check
            for i in range(min(self.mask_first_n, len(tokens))):
                loss_mask[i] = 0
            for i in range(self.mask_first_n, len(tokens)):
                if attention_mask[i] == 1:
                    loss_mask[i] = 1
        else:
            # Normal mode: mask before '=' 
            if eq_pos >= 0:
                for i in range(eq_pos + 1, len(tokens)):
                    if attention_mask[i] == 1:
                        loss_mask[i] = 1
            else:
                # If no '=' found, unmask all real tokens
                loss_mask = attention_mask[:]
        
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        mask = torch.tensor(loss_mask[1:], dtype=torch.float)
        
        return x, y, mask

# -----------------------------------------------------------------------------
# Simple character-level tokenizer
class CharTokenizer:
    def __init__(self, chars):
        self.chars = sorted(list(set(chars)))
        self.vocab_size = len(self.chars) + 1  # +1 for padding
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.pad_token = self.vocab_size - 1
    
    def encode(self, text):
        return [self.char_to_idx.get(ch, self.pad_token) for ch in text]
    
    def decode(self, indices):
        return ''.join([self.idx_to_char.get(idx, '') for idx in indices if idx != self.pad_token])

# -----------------------------------------------------------------------------
# Training functions
def get_batch(dataloader_iter, dataloader, device):
    try:
        x, y, mask = next(dataloader_iter)
    except StopIteration:
        dataloader_iter = iter(dataloader)
        x, y, mask = next(dataloader_iter)
    
    x = x.to(device)
    y = y.to(device)
    mask = mask.to(device)
    
    return x, y, mask, dataloader_iter

@torch.no_grad()
def estimate_loss(model, val_loader, ctx, device, config):
    model.eval()
    losses = []
    accuracies = []
    
    for x, y, mask in val_loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        
        with ctx:
            logits = model(x)
            
            # Only compute loss on masked positions
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                y.view(-1), 
                reduction='none'
            )
            loss = (loss * mask.view(-1)).sum() / (mask.sum() + 1e-8)  # Add epsilon to avoid division by zero
            
            # Compute accuracy on masked positions
            pred = torch.argmax(logits, dim=-1)
            correct = (pred == y) * mask
            accuracy = correct.sum() / (mask.sum() + 1e-8)  # Add epsilon to avoid division by zero
            
        losses.append(loss.item())
        accuracies.append(accuracy.item())
    
    model.train()
    return np.mean(losses), np.mean(accuracies)

# -----------------------------------------------------------------------------
# Main training function
def train(args):
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device setup
    device = args.device
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    # Load data
    print(f"Loading data from {args.data_dir}")
    with open(os.path.join(args.data_dir, 'train.txt'), 'r') as f:
        train_data = f.read().strip().split('\n')
    with open(os.path.join(args.data_dir, 'val.txt'), 'r') as f:
        val_data = f.read().strip().split('\n')
    
    # Create tokenizer
    all_chars = ''.join(train_data + val_data)
    tokenizer = CharTokenizer(all_chars)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Create datasets and dataloaders
    train_dataset = AlgorithmicDataset(train_data, tokenizer, args.block_size, args.mask_first_n)
    val_dataset = AlgorithmicDataset(val_data, tokenizer, args.block_size, args.mask_first_n)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Model initialization
    model_args = dict(
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        block_size=args.block_size,
        bias=args.bias,
        vocab_size=tokenizer.vocab_size,
        dropout=args.dropout
    )
    
    print(f"Initializing {args.n_layer}-layer model")
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    model.to(device)
    
    # Initialize optimizer
    optimizer = model.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        betas=(args.beta1, args.beta2),
        device_type=device_type
    )
    
    # Initialize GradScaler for mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
    
    # Training loop
    train_loader_iter = iter(train_loader)
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    print(f"Starting training for {args.max_steps} steps")
    
    for step in range(args.max_steps):
        # Get batch
        x, y, mask, train_loader_iter = get_batch(train_loader_iter, train_loader, device)
        
        # Forward pass
        with ctx:
            logits = model(x)
            
            # Compute loss only on masked positions
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                y.view(-1), 
                reduction='none'
            )
            loss = (loss * mask.view(-1)).sum() / (mask.sum() + 1e-8)  # Add epsilon to avoid division by zero
            
            # Compute accuracy
            with torch.no_grad():
                pred = torch.argmax(logits, dim=-1)
                correct = (pred == y) * mask
                accuracy = correct.sum() / (mask.sum() + 1e-8)  # Add epsilon to avoid division by zero
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Gradient clipping
        if args.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        
        # Logging
        if step % args.log_interval == 0:
            train_losses.append(loss.item())
            train_accuracies.append(accuracy.item())
            print(f"Step {step}: train loss {loss.item():.4f}, train acc {accuracy.item():.4f}")
        
        # Validation
        if step % args.eval_interval == 0:
            val_loss, val_acc = estimate_loss(model, val_loader, ctx, device, args)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            print(f"Step {step}: val loss {val_loss:.4f}, val acc {val_acc:.4f}")
            
            # Optionally save periodic checkpoint
            if args.save_periodic and step > 0 and step % args.save_interval == 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'step': step,
                    'train_losses': train_losses,
                    'train_accuracies': train_accuracies,
                    'val_losses': val_losses,
                    'val_accuracies': val_accuracies,
                    'config': vars(args),
                    'tokenizer_chars': tokenizer.chars
                }
                
                os.makedirs(args.out_dir, exist_ok=True)
                torch.save(checkpoint, os.path.join(args.out_dir, f'ckpt_step_{step}.pt'))
                print(f"Saved checkpoint at step {step}")
    
    print("Training completed!")
    
    # Save final checkpoint
    os.makedirs(args.out_dir, exist_ok=True)
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'step': args.max_steps,
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'config': vars(args),
        'tokenizer_chars': tokenizer.chars
    }
    torch.save(checkpoint, os.path.join(args.out_dir, 'final_model.pt'))
    print(f"Saved final model to {os.path.join(args.out_dir, 'final_model.pt')}")
    
    # Save config
    with open(os.path.join(args.out_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Save training curves
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(0, len(train_losses) * args.log_interval, args.log_interval), train_losses, label='Train Loss')
    plt.plot(range(0, len(val_losses) * args.eval_interval, args.eval_interval), val_losses, label='Val Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(range(0, len(train_accuracies) * args.log_interval, args.log_interval), train_accuracies, label='Train Acc')
    plt.plot(range(0, len(val_accuracies) * args.eval_interval, args.eval_interval), val_accuracies, label='Val Acc')
    plt.xlabel('Step')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'training_curves.png'))
    print(f"Saved training curves to {os.path.join(args.out_dir, 'training_curves.png')}")
    
    return train_losses, train_accuracies, val_losses, val_accuracies

# -----------------------------------------------------------------------------
# Argument parser
def parse_args():
    parser = argparse.ArgumentParser(description='Train transformer on algorithmic tasks')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing train.txt and val.txt')
    parser.add_argument('--out_dir', type=str, default='out', help='Output directory for checkpoints')
    
    # Model arguments
    parser.add_argument('--n_layer', type=int, default=1, help='Number of transformer layers')
    parser.add_argument('--n_head', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--n_embd', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--block_size', type=int, default=32, help='Maximum sequence length')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--bias', type=bool, default=False, help='Use bias in linear layers')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--max_steps', type=int, default=100000, help='Maximum training steps')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--beta1', type=float, default=0.9, help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.95, help='Adam beta2')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping')
    
    # Logging arguments
    parser.add_argument('--log_interval', type=int, default=100, help='Log training metrics every N steps')
    parser.add_argument('--eval_interval', type=int, default=1000, help='Evaluate on validation set every N steps')
    parser.add_argument('--save_periodic', action='store_true', help='Save periodic checkpoints during training')
    parser.add_argument('--save_interval', type=int, default=10000, help='Save checkpoint every N steps (if save_periodic is True)')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--mask_first_n', type=int, default=0, help='Mask loss on first N tokens (for sanity check)')
    
    return parser.parse_args()

# -----------------------------------------------------------------------------
if __name__ == '__main__':
    args = parse_args()
    train(args)