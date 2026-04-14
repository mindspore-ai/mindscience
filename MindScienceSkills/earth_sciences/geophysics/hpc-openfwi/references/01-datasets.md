# OpenFWI Datasets

## Overview

OpenFWI provides 12 benchmark datasets (2.1 TB total) for seismic FWI research.

## Dataset Categories

### Velocity Family
| Dataset | Description | Size |
|---------|-------------|------|
| Vel-1.0 | Simple layered velocity | 17 GB |
| Vel-2.0 | Complex layered velocity | 17 GB |
| Vel-3.0 | Multi-scale velocity | 17 GB |

### Fault Family
| Dataset | Description | Size |
|---------|-------------|------|
| Fault-1.0 | Single fault | 17 GB |
| Fault-2.0 | Multiple faults | 17 GB |
| Fault-3.0 | Complex fault network | 17 GB |

### Style Family
| Dataset | Description | Size |
|---------|-------------|------|
| Style-A | Geological style A | 17 GB |
| Style-B | Geological style B | 17 GB |

### Kimberlina Family
| Dataset | Description | Size |
|---------|-------------|------|
| Kimberlina-A | Carbonate reservoir | 17 GB |
| Kimberlina-B | Complex carbonate | 17 GB |

## Data Format

### HDF5 Structure
```
data.h5
├── seismic     # Shape: (N, n_shots, n_time, n_receivers)
└── velocity    # Shape: (N, n_z, n_x)
```

### Typical Dimensions
- N: Number of samples (10,000 - 50,000)
- n_shots: Number of shot gathers (5-11)
- n_time: Time samples (1000)
- n_receivers: Number of receivers (70)
- n_z, n_x: Velocity grid (70 x 70)

## Data Loading

```python
import h5py
import numpy as np

# Load data
with h5py.File('data.h5', 'r') as f:
    seismic = f['seismic'][:]
    velocity = f['velocity'][:]

print(f'Seismic shape: {seismic.shape}')
print(f'Velocity shape: {velocity.shape}')
```

## Data Preprocessing

### Normalization
```python
# Min-max normalization
seismic_norm = (seismic - seismic.min()) / (seismic.max() - seismic.min())
velocity_norm = (velocity - velocity.min()) / (velocity.max() - velocity.min())
```

### Standardization
```python
# Z-score standardization
seismic_std = (seismic - seismic.mean()) / seismic.std()
velocity_std = (velocity - velocity.mean()) / velocity.std()
```
