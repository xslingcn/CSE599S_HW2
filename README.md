# CSE 599S Homework 2
Here is the `README.docx` content transformed into a GitHub-flavored Markdown `README.md` file.

Team Members: [Bohan Fang, Shanli Xing]
All pictures are in the folder: out

## Features

* [cite_start]**Transformer Model**: Utilizes a transformer model adapted from nanoGPT. [cite: 3]
* [cite_start]**Algorithmic Tasks**: Trains models on modular addition, subtraction, and division. [cite: 1]
* **Custom Dataset Handling**: Implements a custom `AlgorithmicDataset` that masks the loss function to compute loss only on the answer tokens (the characters after the '=' sign).
* [cite_start]**Character-Level Tokenizer**: Employs a simple tokenizer designed for mathematical expressions. [cite: 6]
* [cite_start]**Grokking Analysis**: Investigates the grokking phenomenon, particularly in the context of modular division. [cite: 1]
* **Experiment Automation**: Includes scripts for running experiments with multiple seeds and for automated plotting of training curves.

## Project Structure

```
.
[cite_start]├── model.py              # Transformer model (adapted from nanoGPT) [cite: 3]
[cite_start]├── train.py              # Main training script [cite: 3]
[cite_start]├── inference.py          # Text generation and model evaluation [cite: 3]
[cite_start]├── generate_data.py      # Data generation for modular arithmetic [cite: 3]
[cite_start]├── run_experiments.py    # Script for running multiple seeds [cite: 3]
[cite_start]├── data/                 # Generated datasets [cite: 3]
[cite_start]│   ├── sanity_check/     # Sanity check data [cite: 4]
[cite_start]│   └── algorithmic/      # Modular arithmetic datasets [cite: 4]
│       ├── add_mod97/
│       ├── subtract_mod97/
│       ├── divide_mod97/
│       └── ...
[cite_start]└── out/                  # Model checkpoints and results [cite: 3]
```

## Setup and Requirements

### Dependencies
* Python 3.8+
* PyTorch 2.0+
* `numpy`
* `matplotlib`
* A CUDA-capable GPU is recommended but not required.

Install dependencies using pip:
```bash
pip install torch numpy matplotlib
```

## How to Run

### 1. Generate Datasets
First, generate the datasets for the algorithmic tasks and sanity checks.

**Modular Arithmetic Data:**
[cite_start]The following command generates datasets for addition, subtraction, and division for moduli 97 and 113. [cite: 5]
```bash
python generate_data.py --operations add,subtract,divide --moduli 97,113 --output_dir data/algorithmic
```
* **Dataset Details**:
    * [cite_start]For a prime modulus `p`, all possible equations are generated for addition (`a + b`), subtraction (`a - b`), and division (`a / b`). [cite: 7]
    * [cite_start]For division, the equation `a/b=c` is generated such that `a ≡ b × c (mod p)`. [cite: 8]
    * The datasets are split into 70% for training, 15% for validation, and 15% for testing.

**Sanity Check Data:**
This command generates a simple dataset for sanity checks.
```bash
python generate_data.py --sanity_check --output_dir data/sanity_check
```

### 2. Run Training
Train the transformer model on a generated dataset. The example below trains a model on addition modulo 97.

```bash
python train.py \
    --data_dir data/algorithmic/add_mod97 \
    --out_dir out/add_mod97 \
    --n_layer 1 \
    --n_embd 128 \
    --n_head 4 \
    --batch_size 64 \
    --max_steps 100000
```

### 3. Run Inference
To generate answers for new prompts using a trained model, run the inference script.

```bash
python inference.py --checkpoint out/add_mod97/final_model.pt --prompts "23+45=" --temperature 0.1
```

## Sanity Checks
To ensure the model's basic learning capabilities, two sanity checks were performed:
1.  **Basic Memorization**: The model was trained to memorize the string "I love machine learning."
2.  **Masked Memorization**: The model was trained on the same string, but the loss on the first three tokens was masked.

Both tests passed successfully, with the training loss approaching zero.

## Results and Analysis

### Addition and Subtraction
* **Configuration**: Models with 1-2 layers, an embedding size of 128, and 4 heads were trained for 100,000 steps.
* [cite_start]**Outcome**: The models successfully learned both modular addition and subtraction, achieving high test accuracy. [cite: 9]

### Division and Grokking
The task of learning modular division proved more challenging and led to an investigation of the grokking phenomenon.

* [cite_start]**Initial Challenges**: Using smaller batch sizes (e.g., 16, 64) with multiple random seeds, the model failed to learn the division task. [cite: 10]
* [cite_start]**Breakthrough**: After increasing the batch size to 256 and 512, the model successfully learned, achieving high accuracy on both the training and validation sets. [cite: 10] This delayed generalization is known as grokking.

### Ablation Study: The Impact of Batch Size

[cite_start]An ablation study was conducted to understand how batch size affects grokking on the modular division task. [cite: 11]

* **Key Findings**:
    * [cite_start]Larger batch sizes (256, 512) were found to increase the likelihood of the model learning and exhibiting grokking. [cite: 12]
    * [cite_start]Smaller batch sizes (≤64) prevented the model from learning within the 100,000 training steps. [cite: 13]

* **Hypothesis**:
    It is hypothesized that larger batch sizes contribute to learning by providing more stable gradients, a better signal-to-noise ratio for discovering algebraic structures, and a sufficient number of examples per update to recognize patterns across the entire modular space.
