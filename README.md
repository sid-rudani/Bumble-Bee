# BumbleBee
![banner](Bee.png)
A from-scratch decoder-only Transformer language model built in PyTorch, based on a Tiny Stories-style character-level example.

## What this project contains

- `BumbleBee.ipynb` - a Jupyter notebook that walks through:
  - loading and encoding a character-level TinyStories dataset
  - implementing tokenization with character-to-index mappings
  - training a simple bigram language model
  - explaining the self-attention mechanism with toy examples
  - building a full Transformer decoder block from scratch
  - training the final `BumbleBee` model and generating text

I tried to annotate the notebook so that the reader can have a better understanding (Of course it includes me also in a few years when I look back at this notebook). The notebook is designed to be educational and to help understand the inner workings of a Transformer language model.
## Dataset

The notebook downloads the TinyStories validation dataset from Hugging Face:

- `https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt`

The dataset is processed at the character level, with a vocabulary built from the unique characters in the file.

## Model overview

The final model is a small autoregressive Transformer language model with:

- character-level token embeddings
- positional embeddings
- multiple Transformer blocks
- multi-head self-attention with causal masking
- feed-forward network inside each block
- layer normalization and dropout
- output linear projection to vocabulary logits

### Final hyperparameters

- `block_size = 128`
- `batch_size = 32`
- `max_iters = 5000`
- `learning_rate = 1e-3`
- `n_embd = 64`
- `n_head = 8`
- `n_layer = 8`
- `dropout = 0.2`

## Training

The notebook trains the model using `AdamW` on batches sampled from the training split. It also evaluates train and validation loss periodically using an `estimate_loss()` helper.

## Generation

After training, the model can generate new text autoregressively from a starting context token.

## Usage

1. Install Python and PyTorch.
2. Open `BumbleBee.ipynb` in Jupyter or VS Code.
3. Run the notebook cells in order.
4. The notebook downloads the dataset, prepares batches, builds the model, trains it, and then generates sample text.

## Requirements

- Python 3.8+
- PyTorch
- Jupyter Notebook / JupyterLab or VS Code Notebook support

## Notes

- This is an educational implementation, not intended for production.
- The notebook includes exploratory sections on attention, matrix multiplication, and layer normalization.
- The smaller TinyStories dataset is used because of hardware constraints.
