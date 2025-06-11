import os
import json
import argparse
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# Tokenizer class (must match the one used in training)
class NumberTokenizer:
    def __init__(self, tokens_list):
        """Initialize tokenizer from a list of tokens or equations."""
        # If tokens_list is a list of equations, extract tokens
        if tokens_list and isinstance(tokens_list[0], str) and any(op in tokens_list[0] for op in '+-*/='):
            # Extract all unique tokens from equations
            tokens = set()
            
            for equation in tokens_list:
                # Parse equation to extract numbers and operators
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
                
                # Don't forget the last number
                if current_num:
                    tokens.add(current_num)
            
            # Sort tokens: operators first, then numbers sorted numerically
            operators = ['+', '-', '*', '/', '=']
            numbers = sorted([t for t in tokens if t.isdigit()], key=int)
            
            # Build vocabulary
            self.tokens = operators + numbers
        else:
            # tokens_list is already a list of tokens
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
        
        # Don't forget the last number
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

# -----------------------------------------------------------------------------
# Generation function
@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens, temperature, top_k, device):
    """Generate text from a prompt"""
    model.eval()
    
    # Encode the prompt
    if prompt:
        tokens = tokenizer.encode(prompt)
        x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    else:
        # Start with a random token if no prompt
        x = torch.randint(0, tokenizer.vocab_size - 1, (1, 1), device=device)
    
    # Generate tokens
    for _ in range(max_new_tokens):
        # Crop context if it's getting too long
        x_cond = x if x.size(1) <= model.config.block_size else x[:, -model.config.block_size:]
        
        # Get predictions
        logits = model(x_cond)
        
        # Focus on the last time step
        logits = logits[:, -1, :] / temperature
        
        # Optionally crop probabilities to only top-k
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # Append to the sequence
        x = torch.cat((x, idx_next), dim=1)
    
    # Decode the generated sequence
    generated_tokens = x[0].tolist()
    return tokenizer.decode(generated_tokens)

# -----------------------------------------------------------------------------
# Main inference function
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
    tokenizer_chars = checkpoint.get('tokenizer_chars', None)
    if tokenizer_chars is None:
        # Try to reconstruct tokenizer from data
        data_dir = checkpoint.get('config', {}).get('data_dir', None)
        if data_dir:
            train_file = os.path.join(data_dir, 'train.txt')
            with open(train_file, 'r') as f:
                train_data = f.read().strip().split('\n')
            # Create tokenizer from data
            tokenizer = NumberTokenizer(train_data)
        else:
            # Fall back to a default tokenizer for arithmetic
            default_tokens = ['+', '-', '*', '/', '='] + [str(i) for i in range(200)]
            tokenizer = NumberTokenizer(default_tokens)
    else:
        tokenizer = NumberTokenizer(tokenizer_chars)
    print(f"Loaded tokenizer with vocabulary size: {tokenizer.vocab_size}")
    
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
    
    # Test on specific examples if provided
    if args.test_examples:
        print("\nTesting on specific examples:")
        test_examples = [
            "0+0=", "1+1=", "5+7=", "23+45=", "99+1=",
            "10-5=", "7-3=", "100-1=",
            "10/5=", "20/4=", "100/10="
        ]
        
        correct = 0
        total = 0
        
        for example in test_examples:
            # Skip if example doesn't match our task
            if not any(op in example for op in ['+', '-', '/']):
                continue
            
            with ctx:
                generated = generate(
                    model, tokenizer, example,
                    max_new_tokens=10,
                    temperature=0.1,  # Low temperature for deterministic output
                    top_k=1,
                    device=device
                )
            
            # Extract the generated answer
            generated_answer = generated[len(example):].strip()
            
            print(f"{example} -> {generated_answer}")
            
            # For simple validation (you might want to implement proper checking)
            total += 1
    
    print("\nInference completed!")

# -----------------------------------------------------------------------------
# Argument parser
def parse_args():
    parser = argparse.ArgumentParser(description='Generate text from trained transformer model')
    
    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    
    # Generation arguments
    parser.add_argument('--prompts', type=str, nargs='+', default=None, help='Prompts for generation')
    parser.add_argument('--max_new_tokens', type=int, default=1, help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling (None for no restriction)')
    
    # Mode arguments
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('--test_examples', action='store_true', help='Test on predefined examples')
    
    # Other arguments
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    
    return parser.parse_args()

# -----------------------------------------------------------------------------
if __name__ == '__main__':
    args = parse_args()
    main(args)