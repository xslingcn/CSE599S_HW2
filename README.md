# CSE 599S Homework 2

**Team Members**: [Bohan Fang, Shanli Xing]

## Overview

This repository contains our implementation of transformer models for learning modular arithmetic operations and studying the grokking phenomenon. We train small transformers from scratch on algorithmic tasks (addition, subtraction, and division modulo p) and analyze their learning dynamics.

## Repository Structure

```
.
├── model.py              # Transformer model definition (adapted from nanoGPT)
├── train.py              # Main training script with loss masking
├── inference.py          # Text generation and model evaluation
├── generate_data.py      # Data generation for modular arithmetic
├── data/                 # Generated datasets
│   ├── sanity_check/     # Sanity check data
│   └── algorithmic/      # Modular arithmetic datasets
└── out/                  # Model checkpoints, trainning params and plots
```

## How to Run the Code

### 1. Generate Data

```bash
# Generate all modular arithmetic datasets
python generate_data.py --operations add,subtract,divide --moduli 97,113 --output_dir data/algorithmic/

# Generate sanity check data
python generate_data.py --sanity_check --output_dir data/sanity_check/

```

### 2. Train Models

```bash
# Train on addition (example)
python train.py --data_dir data/algorithmic/add_mod113 --out_dir out/add_mod113_layer2_seed456 --seed 456 --n_layer 2 --n_embd 128 --n_head 4 --batch_size 64 --max_steps 100000 --learning_rate 0.001 --log_interval 1000 --eval_points_per_decade 16
```

### 3. Run Inference

```bash
python inference.py --checkpoint out/divide_mod97_layer2_seed42_batch512/final_model.pt --prompts "91/7=" --max_new_tokens 1
```

### Experiment Script

Or you can use the provided `run_experiment.py` to view and run experiments.

```bash
python run_experiment.py --list
```

---

## Deliverables

View [report.pdf](https://github.com/xslingcn/CSE599S_HW2/blob/main/report.pdf).
