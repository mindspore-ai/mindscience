# PowerFlowNet - MindSpore Implementation

A complete PowerFlowNet in MindSpore version, supporting both CPU and Ascend NPU devices.

[中文版本](README.md)

## Overview

PowerFlowNet leverages Message Passing Graph Neural Networks (GNNs) for high-quality power flow approximation. This repository provides a MindSpore version, featuring 11 GNN architecture variants and a complete data processing pipeline.

## Features

✅ **Self-Implemented** - Zero external GNN library dependency (MessagePassing, TAGConv, degree)  
✅ **Ascend NPU Optimized** - Optimized for Huawei Ascend hardware, efficient PYNATIVE_MODE execution  
✅ **11 Model Architectures** - MLP, GCN, MPN and 8 variants  
✅ **Dual Data Format Support** - PowerFlowData (12D) and PowerFlowDataV2 (4D, recommended)  
✅ **Fully Verified** - Complete alignment tests and numerical stability verification  
✅ **Apache 2.0 License** - Legal derivative of the original MIT version  

## Project Structure

```text
powerflownet/
├── src/                        # Core source code
│   ├── __init__.py            # Package exports (MPN, PowerFlowData, PowerFlowDataV2)
│   ├── argument_parser.py     # Argument parsing (JSON config + CLI)
│   ├── gnn_ops.py             # GNN operations (MessagePassing, TAGConv, degree)
│   ├── cpu_npu_ops.py         # CPU/Ascend compatibility layer
│   ├── data_utils.py          # Data utilities (Data, DataLoader, InMemoryDataset)
│   ├── power_flow_data.py     # Power flow data processing (5 classes, 2 formats)
│   ├── mpn.py                 # Message Passing Networks (9 MPN variants)
│   ├── gcn.py                 # Graph Convolutional Networks (GCN, SkipGCN)
│   ├── mlp.py                 # MLP baseline model
│   ├── training.py            # Training utilities and callbacks
│   ├── evaluation.py          # Evaluation metrics and validation
│   ├── custom_loss_functions.py # Custom loss functions
│   └── __pycache__/           # Python cache
├── configs/                    # Configuration files
│   └── config.py              # Device configuration and MindSpore initialization
├── data/                       # Data directory
│   └── mindspore/             # MindSpore format data (processed and raw)
├── models/                     # Saved model checkpoints
│   ├── 14/                    # 14-bus system models
│   └── 14v2/                  # 14-bus system V2 format models
├── logs/                       # Training logs and results
│   ├── 14/                    # Training logs for 12D data format
│   └── 14v2/                  # Training logs for 4D V2 format (recommended)
├── README.md                  # Chinese documentation
├── README_EN.md               # English documentation
├── README_MINDSPORE_MIGRATION.md # Detailed migration guide
├── train.py                   # Training script (original 12D format)
├── test.py                    # Evaluation script
├── requirements.txt           # Python dependencies
└── LICENSE                    # Apache 2.0 License
```

## Quick Start

### Installation

```bash
# Create conda environment
conda create -n mind python=3.9
conda activate mind

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# MLP baseline (fast)
python train.py --model mlp --case 14 --epochs 20

# MPN Message Passing Network (recommended)
python train.py --model mpn --case 14 --epochs 20

# GCN Graph Convolutional Network
python train.py --model gcn --case 14 --epochs 20

# Using V2 data format (recommended, 4D input)
# Switch to PowerFlowDataV2 in train.py
```

### Evaluation

```bash
# Evaluate trained model
python test.py --model mlp --run_id <run_id>
```

## Supported Models

### Base Models (3 types)

| Model | Description | Parameters |
|-------|-------------|------------|
| `mlp` | Multi-Layer Perceptron baseline | Small |
| `gcn` | Graph Convolutional Network | Medium |
| `mpn` | Message Passing Network | Medium |

### MPN Variants (8 types)

| Model | Description |
|-------|-------------|
| `skip_mpn` | MPN with skip connections |
| `mask_embed_mpn` | MPN with mask embedding |
| `multi_mpn` | Multi-step MP + convolution |
| `mask_embed_multi_mpn` | Mask embedding + multi-step MP |
| `mask_embed_multi_mpn_nomp` | Mask embedding + multi-step conv (no MP) |
| `mpn_simplenet` | Simplified MPN |
| `multi_conv_net` | Multi-parallel convolutions |

## Data Formats

### V2 Format (Recommended, 4D input)

Optimized and recommended format for Ascend NPU:

```text
├── node_features.npy      # (N_samples, N_nodes, 4) - Normalized power
├── edge_features.npy      # (N_samples, N_edges, 2) - Impedance
└── edge_index.npy         # (2, N_edges) - Edge connectivity
```

### Original Format (12D input)

Data input format:

```text
├── node_features.npy      # (N_samples, N_nodes, 9) - one-hot + features
├── edge_features.npy      # (N_samples, N_edges, 7) - Multiple edge attributes
└── edge_index.npy         # (2, N_edges) - Edge connectivity
```

Download datasets from: [Surf Drive Link](https://surfdrive.surf.nl/files/index.php/s/Qw4RHLvI2RPBIBL)

## Key Features

### 1. Self-Implemented GNN Operations

- **MessagePassing**: Generic GNN base class supporting custom aggregation
- **TAGConv**: Topology-aware graph convolution with k-hop neighborhood aggregation
- **degree function**: Compute node degree in graphs, supports weighted degree

### 2. Ascend NPU Optimization

- PYNATIVE_MODE + JIT level O0 for Ascend compatibility
- CPU/Ascend compatible operation layer (gather, scatter, where)
- No forced distributed mode (RANK_TABLE_FILE removed)

### 3. Data Processing Pipeline

- **PowerFlowData**: Flexible multi-format data loading (12D format)
- **PowerFlowDataV2**: Optimized vectorized data processing (4D format)
- **Graph batching**: Merge multiple graphs into single batch
- **Physics constraints**: Normalization and feature constraints

### 4. Complete Training Framework

- Flexible argument parsing (JSON config + CLI)
- Training callbacks and early stopping
- Complete evaluation metrics (MAE, MSE, RMSE, etc.)

## Requirements

- **MindSpore**: >= 2.7.0
- **Python**: 3.9.0
- **NumPy**: >= 1.19.0
- **tqdm**: Progress bar
- **matplotlib**: Optional, for visualization

## License

This project is licensed under the Apache 2.0 License. The code is derived from:

**Original Project**: [PowerFlowNet (ericyangyu/PowerFlowNet)](https://github.com/stavrosorf/poweflownet)

- Original License: MIT License
- Migrated Content: Framework adaptation, data processing, model architecture

**Major Changes**:

- MindSpore framework migration
- Ascend NPU targeted optimization
- Data processing pipeline refactoring and optimization

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{LIN2024110112,
  title = {PowerFlowNet: Power flow approximation using message passing Graph Neural Networks},
  journal = {International Journal of Electrical Power & Energy Systems},
  volume = {160},
  pages = {110112},
  year = {2024},
  issn = {0142-0615},
  doi = {https://doi.org/10.1016/j.ijepes.2024.110112},
  author = {Nan Lin and Stavros Orfanoudakis and Nathan Ordonez Cardenas and Juan S. Giraldo and Pedro P. Vergara},
}
```

## Quick Reference

### Import Models and Data

```python
from src import MPN, PowerFlowDataV2
from src.data_utils import DataLoader

# Load model
model = MPN(nfeature_dim=4, efeature_dim=2, output_dim=4,
            hidden_dim=64, n_gnn_layers=3, k=3, dropout_rate=0.1)

# Load data
dataset = PowerFlowDataV2(data_path='data/mindspore', case=14)
loader = DataLoader(dataset, batch_size=32)
```

### Training Loop

```python
import mindspore as ms
from mindspore import nn

optimizer = nn.optim.Adam(model.trainable_params(), learning_rate=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(20):
    for batch in loader:
        def forward_fn(data):
            pred = model(data)
            loss = loss_fn(pred, data.y)
            return loss

        loss, grads = ms.value_and_grad(forward_fn, weights=model.trainable_params())(batch)
        optimizer(grads)
```

## Troubleshooting

### Issue: Ascend Compilation Error

**Symptom**: `RuntimeError: Can not find kernel tensor for node`  
**Solution**: Ensure correct device mode in config.py:

```python
ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, jit_config=ms.JitConfig(jit_level="O0"))
```

### Issue: Out of Memory

**Symptom**: OOM error  
**Solution**: Reduce batch size or use PowerFlowDataV2 (more memory efficient)

### Issue: Data Loading Failed

**Symptom**: File not found  
**Solution**: Ensure data files are in `data/mindspore/processed/` directory

## Documentation Resources

- [src/argument_parser.py](src/argument_parser.py) - Argument parsing documentation.
- [src/mpn.py](src/mpn.py) - MPN architecture explanation.
- [src/power_flow_data.py](src/power_flow_data.py) - Data processing details.
