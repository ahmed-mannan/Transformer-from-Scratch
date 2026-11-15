
# Transformer from Scratch

A PyTorch implementation of the original **Encoder–Decoder Transformer** architecture, built to explore how Transformer models work under the hood.

This project implements the core components of the Transformer architecture, including token embeddings, positional encodings, multi-head attention, encoder and decoder blocks, and autoregressive text generation.

It also includes a complete training pipeline for an English-to-Italian machine translation task using the OPUS Books dataset.

## Overview

The goal of this project is to understand and implement the Transformer architecture without relying on PyTorch's built-in Transformer layers.

The implementation covers the main building blocks of the architecture and connects them into a complete encoder–decoder model.

**Project details:**
- Framework: PyTorch
- Task: Neural Machine Translation
- Source language: English
- Target language: Italian
- Dataset: OPUS Books
- Tokenization: WordLevel Tokenizer

## Architecture

The model follows the encoder–decoder architecture introduced in the paper *Attention Is All You Need*.

### Core components

| Component | Description |
|---|---|
| Input Embeddings | Converts token IDs into dense vector representations. |
| Positional Encoding | Adds positional information using sinusoidal functions. |
| Multi-Head Attention | Allows the model to focus on different parts of a sequence. |
| Feed-Forward Network | Applies transformations to each position independently. |
| Layer Normalization | Normalizes intermediate representations. |
| Residual Connections | Helps information flow through the network. |
| Encoder | Processes the source sequence and generates contextual representations. |
| Decoder | Generates the target sequence using masked self-attention and cross-attention. |
| Projection Layer | Converts decoder outputs into vocabulary logits. |

### Default model configuration

| Parameter | Value |
|---|---:|
| Batch Size | 8 |
| Epochs | 20 |
| Learning Rate | 0.0001 |
| Sequence Length | 350 |
| Model Dimension | 512 |
| Encoder Layers | 6 |
| Decoder Layers | 6 |
| Attention Heads | 8 |
| Feed-Forward Dimension | 2048 |
| Dropout | 0.1 |

## Project Structure

```text
Transformer-from-Scratch/
│
├── config.py
├── dataset.py
├── model.py
├── train.py
├── test_transformer.py
└── README.md
```

| File | Purpose |
|---|---|
| `model.py` | Implements the complete Transformer architecture and its components. |
| `dataset.py` | Handles bilingual data preparation, padding, and attention masks. |
| `config.py` | Stores model and training configurations. |
| `train.py` | Handles tokenization, dataset loading, training, validation, and checkpoint saving. |
| `test_transformer.py` | Tests model components, dataset processing, decoding, and checkpoint functionality. |

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/ahmed-mannan/Transformer-from-Scratch.git

cd Transformer-from-Scratch
```

### 2. Create a virtual environment

**Windows:**

```powershell
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

Install PyTorch according to your system configuration using the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).

Install the remaining dependencies:

```bash
pip install datasets tokenizers tqdm tensorboard torchmetrics
```

### 4. Run tests

```bash
python test_transformer.py
```

The test script checks:

- Causal attention masking
- Bilingual dataset preparation
- Tensor dimensions and shapes
- Transformer forward and backward passes
- Greedy decoding
- Model checkpoint saving and loading
- Validation pipeline

### 5. Train the model

```bash
python train.py
```

The training script automatically:

1. Downloads the configured OPUS Books dataset.
2. Builds or loads source and target tokenizers.
3. Splits the dataset into training and validation sets.
4. Converts sentences into token sequences.
5. Creates attention masks.
6. Trains the Transformer using the Adam optimizer.
7. Evaluates translations using greedy decoding.
8. Saves model checkpoints after each epoch.

## Training Configuration

The main training settings can be modified in `config.py`.

```python
{
    "batch_size": 8,
    "num_epochs": 20,
    "lr": 10**-4,
    "seq_len": 350,
    "d_model": 512,
    "datasource": "opus_books",
    "lang_src": "en",
    "lang_tgt": "it"
}
```

You can change the language pair, batch size, learning rate, number of epochs, and sequence length according to your requirements.

## Training Monitoring

Training loss is recorded using TensorBoard.

To view the training logs:

```bash
tensorboard --logdir runs
```

The validation pipeline also supports the following metrics when `torchmetrics` is installed:

- Character Error Rate (CER)
- Word Error Rate (WER)
- BLEU Score

## Generated Files

Running the training script creates additional files and directories:

| Output | Purpose |
|---|---|
| `tokenizer_en.json` | Source language tokenizer |
| `tokenizer_it.json` | Target language tokenizer |
| `opus_books_weights/` | Saved model checkpoints |
| `runs/tmodel/` | TensorBoard training logs |

## References

- Vaswani et al. (2017), [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets)
- [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers)

## About

This is an educational project focused on understanding the internal workings of the Transformer architecture through a practical PyTorch implementation.

**Author:** Ahmed Mannan

[GitHub Profile](https://github.com/ahmed-mannan)

---

If you have suggestions or find an issue, feel free to open an issue or contribute to the project.

