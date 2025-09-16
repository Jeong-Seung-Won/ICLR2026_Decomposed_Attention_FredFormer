# TT-based Time Series Forecasting

A PyTorch implementation of time series forecasting using Tensor Train (TT) decomposition with transformer architecture for efficient multivariate time series prediction.

## Overview

This project implements a novel approach to time series forecasting that combines:
- **Tensor Train (TT) decomposition** for parameter-efficient linear transformations
- **Frequency domain processing** using RFFT/IRFFT operations
- **Transformer-based architecture** with multi-head attention
- **Channel-wise positional embeddings** for multivariate time series

## Installation

```bash
git clone [<repository-url>](https://github.com/Jeong-Seung-Won/ICLR2026_Decomposed_Attention_FredFormer)
cd tt-time-series-forecasting
pip install -r requirements.txt
```

## Model Configuration

The model is configured in `config.py`:

```python
ours_tt_cfg = SimpleNamespace(
    model="ours_tt", 
    enc_in=6,           # Input channels/variables
    seq_len=512,        # Input sequence length
    pred_len=90,        # Prediction horizon
    d_model=45,         # Model dimension
    cf_dim=128,         # Channel feature dimension
    cf_depth=2,         # Transformer depth
    cf_heads=4,         # Number of attention heads
    cf_mlp=128,         # MLP hidden dimension
    cf_head_dim=16,     # Attention head dimension
    cf_drop=0.1,        # Dropout rate
    head_dropout=0.1,   # Head dropout rate
    rank=[1, 3, 32, 1]  # Tensor Train ranks
)
```

## Usage

### Quick Start

```python
from config import MODEL_ZOO, CFG_REGISTRY

# Load configuration
cfg = CFG_REGISTRY["ours_tt"]

# Initialize model
model = MODEL_ZOO["ours_tt"](cfg)

# Forward pass
# Input: [batch_size, seq_len, n_channels]
# Output: [batch_size, pred_len, n_channels]
prediction = model(input_tensor)
```

### Training

```bash
python main.py
```

The training script includes:
- **Early stopping** with patience of 5 epochs
- **Cosine annealing** learning rate scheduling
- **Mixed precision training** with GradScaler
- **Automatic model checkpointing**

### Data Format

The model expects data in the following format:
- **Input**: `[batch_size, seq_len, n_channels]`
- **Output**: `[batch_size, pred_len, n_channels]`

Data should be preprocessed with:
- Normalization using global mean and standard deviation
- Stored as PyTorch tensor files (`.pt`)
- Metadata CSV files listing file paths

## Project Structure

```
├── config.py          # Model configurations and registry
├── DAF.py             # Core model implementation
├── main.py            # Training script
├── utils.py           # Data loading and utility functions
└── README.md          # This file
```

Example output:
```
ours_tt         →  123,456 parameters
Train batches : 1000
Val   batches : 200
Test  batches : 300

[ours_tt] Ep 25/50 | Train MSE 0.0012345 | Val MSE 0.00012345 | 12.34s
Finished ours_tt    → Test MSE: 0.00012345  |  MAE: 0.0012345
```

## Customization

### Adding New Models
1. Implement your model class in `DAF.py`
2. Add configuration to `config.py`
3. Register in `MODEL_ZOO` and `CFG_REGISTRY`

### Hyperparameter Tuning
- Adjust tensor ranks in `rank` parameter
- Modify transformer depth and attention heads
- Tune learning rate and batch size in `main.py`

## Citation

If you use this code in your research, please cite:
```bibtex
@misc{DAF,
  title={Decomposed Attention FredFormer: Large Time-series Prediction Model for Satellite Orbit Prediction},
  author={Seungwon Jeong, Kangjun Lee, Jounu Park, Simon S. Woo, Yujin Shin},
  year={2025},
  url={[https://github.com/your-username/tt-time-series-forecasting](https://github.com/Jeong-Seung-Won/ICLR2026_Decomposed_Attention_FredFormer)}
}
```
For questions or issues, please open an issue on GitHub or contact [bbigaa123@g.skku.edu].
