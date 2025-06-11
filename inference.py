import os
import json
import argparse
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from model import GPTConfig, GPT

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

class NumberTokenizer:
    def __init__(self, tokens_list):
        """Initialize tokenizer from a list of tokens or equations."""
        if tokens_list and isinstance(tokens_list[0], str) and any(op in tokens_list[0] for op in '+-*/='):
            tokens = set()
            
            for equation in tokens_list:
                current_num = ''
                for char in equation:
                    if char.isdigit():
                        current_num += char
                    else:
                        if current_num:
                            tokens.add(current_num)
                            current_num = ''
                        if char in '+-*/=':
                            tokens.add(char)
                
                if current_num:
                    tokens.add(current_num)
            
            operators = ['+', '-', '*', '/', '=']
            numbers = sorted([t for t in tokens if t.isdigit()], key=int)
            
            self.tokens = operators + numbers
        else:
            self.tokens = tokens_list
            
        self.vocab_size = len(self.tokens)
        self.token_to_idx = {token: i for i, token in enumerate(self.tokens)}
        self.idx_to_token = {i: token for i, token in enumerate(self.tokens)}
    
    def encode(self, text):
        """Encode text into token indices."""
        tokens = []
        current_num = ''
        
        for char in text:
            if char.isdigit():
                current_num += char
            else:
                if current_num:
                    if current_num in self.token_to_idx:
                        tokens.append(self.token_to_idx[current_num])
                    current_num = ''
                if char in self.token_to_idx:
                    tokens.append(self.token_to_idx[char])
        
        if current_num and current_num in self.token_to_idx:
            tokens.append(self.token_to_idx[current_num])
        
        return tokens
    
    def decode(self, indices):
        """Decode token indices back to text."""
        result = ''
        for idx in indices:
            if idx in self.idx_to_token:
                result += self.idx_to_token[idx]
        return result

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens, temperature, top_k, device):
    """Generate text from a prompt"""
    model.eval()
    
    if prompt:
        tokens = tokenizer.encode(prompt)
        x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    else:
        x = torch.randint(0, tokenizer.vocab_size - 1, (1, 1), device=device)
    
    for _ in range(max_new_tokens):
        # Crop context if it's getting too long
        x_cond = x if x.size(1) <= model.config.block_size else x[:, -model.config.block_size:]
        
        logits = model(x_cond)
        
        # Focus on the last time step
        logits = logits[:, -1, :] / temperature
        
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        
        probs = F.softmax(logits, dim=-1)
        
        idx_next = torch.multinomial(probs, num_samples=1)
        
        x = torch.cat((x, idx_next), dim=1)
    
    generated_tokens = x[0].tolist()
    return tokenizer.decode(generated_tokens)

def main(args):
    # Set device
    device = args.device
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Load tokenizer
    tokenizer_type = checkpoint.get('tokenizer_type', 'number')  # Default to number for backward compatibility
    tokenizer_chars = checkpoint.get('tokenizer_chars', None)
    
    if tokenizer_chars is None:
        data_dir = checkpoint.get('config', {}).get('data_dir', None)
        if data_dir:
            train_file = os.path.join(data_dir, 'train.txt')
            with open(train_file, 'r') as f:
                train_data = f.read().strip().split('\n')
            # Check if this is arithmetic data
            sample_data = train_data[0] if train_data else ""
            is_arithmetic = any(op in sample_data for op in ['+', '-', '/', '='])
            
            if is_arithmetic:
                tokenizer = NumberTokenizer(train_data)
            else:
                all_chars = ''.join(train_data)
                tokenizer = CharTokenizer(all_chars)
        else:
            # Fall back to a default tokenizer for arithmetic
            default_tokens = ['+', '-', '*', '/', '='] + [str(i) for i in range(200)]
            tokenizer = NumberTokenizer(default_tokens)
    else:
        # Load tokenizer based on type
        if tokenizer_type == 'char':
            tokenizer = CharTokenizer(tokenizer_chars)
        else:
            tokenizer = NumberTokenizer(tokenizer_chars)
    
    print(f"Loaded {tokenizer_type} tokenizer with vocabulary size: {tokenizer.vocab_size}")
    
    # Initialize model
    model_args = checkpoint['model_args']
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    
    # Load model weights
    state_dict = checkpoint['model']
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    
    print(f"Loaded model with {model_args['n_layer']} layers, {model_args['n_embd']} dimensions")
    
    # Interactive generation loop
    if args.interactive:
        print("\nInteractive mode. Type 'quit' to exit.")
        print("Enter a prompt (e.g., '12+45=' for addition):")
        
        while True:
            prompt = input("\n> ")
            if prompt.lower() == 'quit':
                break
            
            with ctx:
                generated = generate(
                    model, tokenizer, prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    device=device
                )
            
            print(f"Generated: {generated}")
    
    else:
        # Single generation from command line prompt
        prompts = args.prompts if args.prompts else [""]
        
        for i, prompt in enumerate(prompts):
            print(f"\nPrompt {i+1}: '{prompt}'")
            
            with ctx:
                generated = generate(
                    model, tokenizer, prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    device=device
                )
            
            print(f"Generated: {generated}")
            print("-" * 50)
    
    print("\nInference completed!")

def parse_args():
    parser = argparse.ArgumentParser(description='Generate text from trained transformer model')
    
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    
    parser.add_argument('--prompts', type=str, nargs='+', default=None, help='Prompts for generation')
    parser.add_argument('--max_new_tokens', type=int, default=1, help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling (None for no restriction)')
    
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)