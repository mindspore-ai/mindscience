---
name: dpot
description: NVIDIA PhysicsNeMo DPOT (Denoising Pre-trained Operator Transformer) using AFNO layers for solving Navier-Stokes equations and CFD simulations. Use for spatio-temporal PDE solving, surrogate modeling, and time-series fluid dynamics predictions.
license: Apache 2.0
metadata:
    skill-author: MindSpore Science Team
    model-source: NVIDIA PhysicsNeMo
    github: https://github.com/NVIDIA/physicsnemo/tree/main/examples/cfd/navier_stokes_dpot
    paper: https://arxiv.org/abs/2403.03542
---

# NVIDIA PhysicsNeMo DPOT (Denoising Pre-trained Operator Transformer)

## Overview

DPOT (Denoising Pre-trained Operator Transformer) is a transformer-based neural operator model designed for solving Navier-Stokes equations and computational fluid dynamics (CFD) simulations. It uses Adaptive Fourier Neural Operator (AFNO) layers for efficient spectral mixing and supports both 2D and 3D spatio-temporal predictions.

The model architecture features:
- **Patch Embedding**: Converts input fields into patch tokens with optional coordinate grid concatenation
- **Temporal Aggregation**: MLP or exponential MLP modes for processing temporal sequences
- **AFNO Blocks**: Spectral mixing in Fourier domain with adaptive weighting
- **Flexible Prediction Modes**: Supports `seq2seq` (sequence-to-sequence) and `one2many` (one-to-many) prediction strategies

DPOT is particularly effective for surrogate modeling in CFD applications, enabling fast inference compared to traditional numerical solvers while maintaining accuracy for complex fluid dynamics phenomena.

---

## When to Use

- **CFD Surrogate Modeling**: Replace expensive CFD simulations with fast neural network inference for Navier-Stokes equations
- **Spatio-Temporal PDE Solving**: Solve time-dependent partial differential equations with neural operators
- **Time-Series Fluid Dynamics**: Predict future fluid flow states from historical observations
- **Data-Driven Simulation**: Learn physics from simulation data for rapid prototyping and design optimization
- **Autoregressive Prediction (optional)**: Generate long-term predictions by iteratively feeding outputs back as inputs

---

## How It Works

### 1. Dataset Acquisition and Processing

#### Dataset Requirements

| Requirement | Description |
|-------------|-------------|
| Data Format | HDF5 (.h5) or MATLAB (.mat) files with velocity fields and pressure data |
| Data Size | Default dataset: 5000 samples (train: 4500, test: 500); scalable to larger datasets |
| Data Source | Navier-Stokes simulation data (auto-downloadable from Google Drive) |
| Spatial Resolution | Configurable (default: 64x64 for 2D, 64x64x64 for 3D) |
| Temporal Steps | Variable sequence length (default: 10 time steps) |

#### Data Acquisition Methods

1. **Auto-Download**: Dataset automatically downloads from Google Drive on first run - Default behavior for Navier-Stokes examples
2. **Custom Data**: Prepare HDF5 files with velocity/pressure fields - For custom simulation scenarios
3. **PhysicsNeMo Data Pipeline**: Use PhysicsNeMo's data utilities - For integration with other PhysicsNeMo workflows

#### Data Preprocessing

Users need to preprocess data according to the following steps:

- **Normalization**: Apply min-max or z-score normalization to velocity/pressure fields
- **Temporal Batching**: Organize data into input-output temporal sequences
- **Spatial Gridding (optional)**: Resample data to uniform grid if needed
- **Data Augmentation (optional)**: Apply rotations, flips for increased training diversity

---

### 2. Environment Configuration and Dependencies

#### Dependency Installation

```bash
# Core dependencies
pip install torch>=2.5.0
pip install einops
pip install hydra-core
pip install h5py
pip install scipy
pip install numpy

# PhysicsNeMo installation (recommended)
pip install nvidia-physicsnemo

# Or install from source
git clone https://github.com/NVIDIA/physicsnemo.git
cd physicsnemo
pip install -e .
```

#### Environment Requirements

| Requirement | Specification |
|-------------|---------------|
| Python Version | Python 3.11+ |
| Hardware | NVIDIA GPU (CUDA 11.8+), single GPU currently supported |
| Memory | 16GB+ RAM, 8GB+ VRAM for 2D models, 24GB+ VRAM for 3D models |
| Disk Space | 5GB+ for dataset and model checkpoints |

#### Model Configurations

| Model Type | Parameters | VRAM Requirement | Use Case |
|------------|------------|------------------|----------|
| DPOTNet (2D) | ~30M | ~8GB | 2D Navier-Stokes, planar flows |
| DPOTNet3D | ~100M | ~24GB | 3D Navier-Stokes, volumetric flows |

#### Installation Steps

1. **Step 1**: Install PyTorch with CUDA support - `pip install torch>=2.5.0 --index-url https://download.pytorch.org/whl/cu118`
2. **Step 2**: Install PhysicsNeMo - `pip install nvidia-physicsnemo` or clone from source
3. **Step 3**: Navigate to example directory - `cd physicsnemo/examples/cfd/navier_stokes_dpot`
4. **Step 4**: Verify installation - Run `python -c "import torch; print(torch.cuda.is_available())"`

---

### 3. Usage Limitations and Notes

#### Model Limitations

| Limitation Type | Description |
|----------------|-------------|
| Functional Limitations | Currently single-GPU only; multi-GPU support in development |
| Performance Limitations | Accuracy depends on data quality and coverage |
| Scale Limitations | Memory scales with spatial resolution; 3D models require significant VRAM |
| Training Data | No pre-trained weights currently available; requires training from scratch |

#### Notes

- **Work in Progress**: This example is actively being developed; APIs may change
- **Single GPU**: Currently only single GPU training is supported
- **Training Required**: No pre-trained weights are provided; users must train on their own data
- **Memory Management**: Use gradient checkpointing for large 3D models to reduce memory footprint
- **Numerical Stability**: Monitor for NaN values during training, especially with high learning rates

#### License Agreement

This model uses Apache 2.0 license. The main terms include:

- Free for commercial and non-commercial use
- Modifications must be documented
- No warranty provided

For full license details, see: https://www.apache.org/licenses/LICENSE-2.0

---

### 4. Model Invocation Guide

#### Invocation Process Overview

The basic model invocation process includes:

1. **Model Configuration**: Set up model parameters via Hydra config or direct initialization
2. **Data Loading**: Load HDF5/MATLAB data and create PyTorch DataLoaders
3. **Model Initialization**: Create DPOTNet or DPOTNet3D instance with appropriate settings
4. **Inference**: Run autoregressive prediction for new scenarios

#### Model Initialization

```python
import torch
from physicsnemo.models.dpot import DPOTNet
from physicsnemo.models.dpot.dpot3d import DPOTNet3D

# 2D DPOT Model
model_2d = DPOTNet(
    inp_shape=(64, 64),          # Spatial resolution
    patch_size=(8, 8),           # Patch size for tokenization
    in_channels=3,               # Input channels (e.g., vx, vy, pressure)
    out_channels=3,              # Output channels
    embed_dim=768,               # Embedding dimension
    depth=12,                    # Number of transformer blocks
    num_heads=12,                # Number of attention heads
    mlp_ratio=4.0,               # MLP hidden dimension ratio
    drop_rate=0.0,               # Dropout rate
    attn_drop_rate=0.0,          # Attention dropout rate
    drop_path_rate=0.1,          # DropPath rate for stochastic depth
    temporal_agg="mlp",          # Temporal aggregation: "mlp" or "exp_mlp"
    prediction_mode="seq2seq",   # "seq2seq" or "one2many"
)

# 3D DPOT Model
model_3d = DPOTNet3D(
    inp_shape=(64, 64, 64),      # 3D spatial resolution
    patch_size=(8, 8, 8),        # 3D patch size
    in_channels=4,               # Input channels (e.g., vx, vy, vz, pressure)
    out_channels=4,              # Output channels
    embed_dim=768,
    depth=12
)

# Move to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model_2d.to(device)
```

#### Inference Example

```python
import torch
import numpy as np

def autoregressive_prediction(model, initial_condition, num_steps, device):
    """
    Perform autoregressive prediction for multiple time steps.
    
    Args:
        model: Trained DPOT model
        initial_condition: Initial state tensor (B, T_init, C, H, W)
        num_steps: Number of future steps to predict
        device: torch device
    
    Returns:
        predictions: List of predicted states
    """
    model.eval()
    predictions = []
    current_input = initial_condition.to(device)
    
    with torch.no_grad():
        for step in range(num_steps):
            # Predict next time step
            output = model(current_input)
            predictions.append(output.cpu().numpy())
            
            # Update input for next prediction (sliding window)
            if current_input.shape[1] > 1:
                current_input = torch.cat([
                    current_input[:, 1:, :, :, :],  # Remove oldest frame
                    output.unsqueeze(1)              # Add new prediction
                ], dim=1)
            else:
                current_input = output.unsqueeze(1)
    
    return predictions

# Example usage
model.eval()
initial_state = torch.randn(1, 10, 3, 64, 64)  # Batch=1, 10 time steps, 3 channels, 64x64
predictions = autoregressive_prediction(model, initial_state, num_steps=50, device=device)
print(f"Generated {len(predictions)} future predictions")
```

---

## Reference Resources

### Official Documentation

- [PhysicsNeMo Documentation](https://docs.nvidia.com/deeplearning/physicsnemo/): Official NVIDIA PhysicsNeMo documentation
- [DPOT Example README](https://github.com/NVIDIA/physicsnemo/tree/main/examples/cfd/navier_stokes_dpot): Example-specific documentation
- [AFNO Paper](https://arxiv.org/abs/2011.08898): Adaptive Fourier Neural Operators paper

### Related Tutorials

- [PhysicsNeMo Getting Started](https://docs.nvidia.com/deeplearning/physicsnemo/getting-started.html): Installation and basic usage
- [Neural Operators Guide](https://docs.nvidia.com/deeplearning/physicsnemo/user-guide/neural_operators.html): Guide to neural operator models

### Research Papers

- [DPOT Paper (arXiv:2403.03542)](https://arxiv.org/abs/2403.03542): "Denoising Pre-trained Operator Transformer" - Main research paper describing the model architecture and methodology

### Community Support

- [NVIDIA PhysicsNeMo GitHub](https://github.com/NVIDIA/physicsnemo): Source code and issue tracking
- [NVIDIA Developer Forums](https://forums.developer.nvidia.com/): Community discussions and support