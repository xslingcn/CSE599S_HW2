import os
import random
import argparse
from typing import List, Tuple

def generate_modular_arithmetic_data(
    operation: str, 
    p: int, 
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    Generate data for modular arithmetic operations.
    
    Args:
        operation: One of 'add', 'subtract', 'divide'
        p: The modulus
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        seed: Random seed for reproducibility
    
    Returns:
        train_data, val_data, test_data as lists of equations
    """
    random.seed(seed)
    
    equations = []
    
    if operation == 'add':
        # Generate all possible addition equations
        for a in range(p):
            for b in range(p):
                c = (a + b) % p
                equations.append(f"{a}+{b}={c}")
    
    elif operation == 'subtract':
        # Generate all possible subtraction equations
        for a in range(p):
            for b in range(p):
                c = (a - b) % p
                equations.append(f"{a}-{b}={c}")
    
    elif operation == 'divide':
        # Generate division equations using the same logic as the reference
        # Iterate through all pairs (a, b) from 0 to p-1
        for a in range(p):
            for b in range(1, p):
                c = a  # Store original a value as c
                a = (b * c) % p  # Compute new a value
                equations.append(f"{a}/{b}={c}")
    
    else:
        raise ValueError(f"Unknown operation: {operation}")
    
    # Shuffle the equations
    random.shuffle(equations)
    
    # Split into train, val, test
    n = len(equations)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    
    train_data = equations[:train_size]
    val_data = equations[train_size:train_size + val_size]
    test_data = equations[train_size + val_size:]
    
    return train_data, val_data, test_data

def gcd(a: int, b: int) -> int:
    """Compute greatest common divisor"""
    while b:
        a, b = b, a % b
    return a

def mod_inverse(a: int, m: int) -> int:
    """Compute modular inverse of a modulo m using extended Euclidean algorithm"""
    if gcd(a, m) != 1:
        raise ValueError(f"{a} has no inverse modulo {m}")
    
    # Extended Euclidean Algorithm
    m0, x0, x1 = m, 0, 1
    while a > 1:
        q = a // m
        m, a = a % m, m
        x0, x1 = x1 - q * x0, x0
    
    return x1 + m0 if x1 < 0 else x1

def save_data(data: List[str], filepath: str):
    """Save data to a text file"""
    with open(filepath, 'w') as f:
        f.write('\n'.join(data))
    print(f"Saved {len(data)} equations to {filepath}")

def generate_sanity_check_data(output_dir: str):
    """Generate simple data for sanity checking"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Single string repeated
    train_data = ["I love machine learning"] * 100
    val_data = ["I love machine learning"] * 20
    test_data = ["I love machine learning"] * 20
    
    save_data(train_data, os.path.join(output_dir, 'train.txt'))
    save_data(val_data, os.path.join(output_dir, 'val.txt'))
    save_data(test_data, os.path.join(output_dir, 'test.txt'))
    
    print(f"\nGenerated sanity check data in {output_dir}")

def main(args):
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.sanity_check:
        generate_sanity_check_data(args.output_dir)
        return
    
    # Generate data for each operation and modulus
    operations = args.operations.split(',')
    moduli = [int(p) for p in args.moduli.split(',')]
    
    for operation in operations:
        for p in moduli:
            print(f"\nGenerating {operation} data with modulus {p}")
            
            # Generate the data
            train_data, val_data, test_data = generate_modular_arithmetic_data(
                operation=operation,
                p=p,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                seed=args.seed
            )
            
            # Create directory for this task
            task_dir = os.path.join(args.output_dir, f"{operation}_mod{p}")
            os.makedirs(task_dir, exist_ok=True)
            
            # Save the data
            save_data(train_data, os.path.join(task_dir, 'train.txt'))
            save_data(val_data, os.path.join(task_dir, 'val.txt'))
            save_data(test_data, os.path.join(task_dir, 'test.txt'))
            
            # Print statistics
            print(f"Total equations: {len(train_data) + len(val_data) + len(test_data)}")
            print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
            
            # Save some examples
            print(f"Example equations:")
            for i in range(min(5, len(train_data))):
                print(f"  {train_data[i]}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate data for algorithmic tasks')
    
    parser.add_argument('--output_dir', type=str, default='data', help='Output directory for generated data')
    parser.add_argument('--operations', type=str, default='add,subtract,divide', help='Comma-separated list of operations')
    parser.add_argument('--moduli', type=str, default='97,113', help='Comma-separated list of moduli')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Proportion of data for training')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Proportion of data for validation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--sanity_check', action='store_true', help='Generate sanity check data')
    
    args = parser.parse_args()
    main(args)