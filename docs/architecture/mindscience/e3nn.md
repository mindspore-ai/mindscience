# E3NN: Euclidean Neural Networks

## What is E3NN

E3NN is a Python library based on the MindSpore framework for creating equivariant neural networks. It specializes in handling 3D spatial data, ensuring network consistency under rotational transformations.

**Core Advantages**:

- **Rotational Invariance**: Prediction results remain unchanged after molecular rotation
- **High Data Efficiency**: No need for extensive data augmentation
- **Clear Physical Meaning**: Conforms to the symmetry of physical laws

## Basic Concepts

### Data Representation

E3NN uses **Irreps (Irreducible Representations)** to describe different types of data:

- `0e`: Scalars (such as temperature, energy)
- `1o`: Vectors (such as position, velocity)
- `2e`: Tensors (such as stress)

## Main Features

### 1. Data Representation and Operations

```python
from mindscience.e3nn import o3

# Create irreducible representations
irreps = o3.Irreps("2x0e + 3x1o")  # 2 scalars + 3 vectors
print(irreps.dim)  # Total dimension: 2 + 9 = 11

# Generate random data
x = irreps.randn(-1)
```

### 2. Tensor Product Operations

```python
from mindscience.e3nn import o3

# Combine different features
tp = o3.TensorProduct(
    irreps_in1="2x1o",      # Input1: 2 vectors
    irreps_in2="1x0e",      # Input2: 1 scalar
    irreps_out="2x1o"       # Output: 2 vectors
)
```

### 3. Equivariant Neural Network Layers

```python
from mindscience.e3nn import nn
import mindspore.ops as ops

# Activation function (only for scalars)
act = nn.Activation("3x0e + 2x1o", acts=[ops.tanh, None])

# Gating mechanism
gate = nn.Gate(
    irreps_scalars="8x0e",      # Scalar channels
    acts=[ops.tanh],            # Scalar activation functions
    irreps_gates="8x0e",        # Gating scalars
    act_gates=[ops.sigmoid],    # Gating activation functions
    irreps_gated="8x1o"         # Gated vector channels
)
```

## Library Structure

```text
mindscience.e3nn/
├── o3/                      # Basic operations module
│   ├── irreps.py           # Irreducible representations (Irreps)
│   ├── tensor_product.py   # Tensor product operations
│   ├── spherical_harmonics.py  # Spherical harmonics
│   ├── rotation.py         # Rotation matrices and angle operations
│   ├── wigner.py          # Wigner D matrices
│   ├── norm.py            # Norm calculations
│   └── sub.py             # Sub-representation operations
├── nn/                      # Neural network layers module
│   ├── activation.py       # Equivariant activation functions
│   ├── gate.py            # Gating mechanisms
│   ├── batchnorm.py       # Equivariant batch normalization
│   ├── fc.py              # Fully connected layers (Linear)
│   ├── normact.py         # Normalization-activation combinations
│   ├── one_hot.py         # One-hot encoding
│   └── scatter.py         # Scatter aggregation operations
└── utils/                   # Utility functions module
    ├── batch_dot.py        # Batch dot product operations
    ├── func.py             # General utility functions
    ├── initializer.py      # Parameter initializers
    ├── linalg.py           # Linear algebra tools
    ├── ncon.py             # Tensor network contractions
    ├── perm.py             # Permutation operations
    └── radius.py           # Radius graph construction
```

### Detailed Module Descriptions

#### o3 Module - Basic Mathematical Operations

- **irreps.py**: Core irreducible representation class, defining data types and dimensions
- **tensor_product.py**: Implements equivariant tensor product operations
- **spherical_harmonics.py**: Spherical harmonics calculations
- **rotation.py**: Rotation matrix generation, angle conversion, and other rotation-related operations
- **wigner.py**: Wigner D matrix calculations for rotational transformations
- **norm.py**: Equivariant norm calculations
- **sub.py**: Sub-representation extraction and operations

#### nn Module - Neural Network Layers

- **activation.py**: Equivariant activation functions that can only act on scalar parts
- **gate.py**: Gating mechanisms for controlling vector feature activation
- **batchnorm.py**: Equivariant batch normalization layers
- **fc.py**: Equivariant fully connected layers (Linear)
- **normact.py**: Combined normalization and activation layers
- **one_hot.py**: One-hot encoding utilities
- **scatter.py**: Scatter aggregation operations in graph neural networks

#### utils Module - Utility Functions

- **batch_dot.py**: Efficient batch dot product operations
- **func.py**: Collection of general utility functions
- **initializer.py**: Network parameter initializers
- **linalg.py**: Linear algebra related tools
- **ncon.py**: Tensor network contraction operations
- **perm.py**: Permutation and transposition operations
- **radius.py**: Utility functions for constructing radius graphs

## Core Components

### 1. Irreducible Representations (Irreps)

```python
from mindscience.e3nn import o3

# Create irreducible representations
irreps = o3.Irreps("2x0e + 3x1o + 1x2e")
print(irreps.dim)        # Total dimension: 2 + 9 + 5 = 16
print(irreps.ls)         # Angular momentum quantum numbers: [0, 1, 2]

# Generate data
x = irreps.randn(-1)     # Random tensor
```

### 2. Tensor Product Operations

```python
from mindscience.e3nn import o3

# Fully connected tensor product
tp = o3.TensorProduct(
    irreps_in1="2x1o",      # Input1: vectors
    irreps_in2="1x0e",      # Input2: scalar
    irreps_out="2x1o"       # Output: vectors
)

result = tp(x1, x2)  # 默认weight_mode="inner"，不需要手动提供权重
```

### 3. Equivariant Neural Network Layers

```python
from mindscience.e3nn import nn
import mindspore.ops as ops

# Activation function - can only be used for scalars
act = nn.Activation("3x0e", acts=[ops.tanh])

# Gating mechanism
gate = nn.Gate(
    irreps_scalars="2x0e",      # Scalar channels
    acts=[ops.tanh],            # Scalar activation functions
    irreps_gates="2x0e",        # Gating scalars
    act_gates=[ops.sigmoid],    # Gating activation functions
    irreps_gated="2x1o"         # Gated vector channels
)

# Batch normalization
bn = nn.BatchNorm("2x0e + 3x1o")
```

### 4. Spherical Harmonics

```python
from mindscience.e3nn import o3
import mindspore as ms

# Calculate spherical harmonics
pos = ms.Tensor([[1.0, 0.0, 0.0]])  # Position vector
sh = o3.spherical_harmonics(l=2, x=pos, normalize=True)
```

### 5. Rotation Operations and Wigner D Matrices

```python
from mindscience.e3nn import o3
import mindspore as ms

# Generate rotation matrix
alpha, beta, gamma = 0.1, 0.2, 0.3  # Euler angles
R = o3.angles_to_matrix(alpha, beta, gamma)

# Apply rotation to irreps
irreps = o3.Irreps("1x1o")  # One vector
x = irreps.randn(-1)
D = irreps.wigD_from_matrix(R)  # Wigner D matrix
x_rotated = D @ x  # Rotated vector
```

### 6. Equivariant Linear Layers

```python
from mindscience.e3nn import o3

# Create equivariant linear layer
linear = o3.Linear(
    irreps_in="2x0e + 1x1o",   # Input: 2 scalars + 1 vector
    irreps_out="1x0e + 2x1o"   # Output: 1 scalar + 2 vectors
)

# Forward pass
x = o3.Irreps("2x0e + 1x1o").randn(-1)
y = linear(x)
```

### 7. Batch Normalization

```python
from mindscience.e3nn import nn

# Equivariant batch normalization
bn = nn.BatchNorm("4x0e + 2x1o")

# Apply normalization
x = o3.Irreps("4x0e + 2x1o").randn(32, -1)  # Batch of 32
x_normalized = bn(x)
```

### 8. Norm Calculations

```python
from mindscience.e3nn import o3

# Calculate norms for different irreps
irreps = o3.Irreps("2x0e + 3x1o")
x = irreps.randn(-1)

# Compute norm using Norm class
norm_layer = o3.Norm(irreps)
norm_result = norm_layer(x)
```

### 9. Scatter Aggregation

```python
from mindscience.e3nn import nn
import mindspore as ms

# Scatter aggregation for graph neural networks
scatter = nn.Scatter(mode="add")  # 支持的模式: 'add', 'sum', 'div', 'max', 'min', 'mul'

# Example usage
src = ms.Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # Source features
index = ms.Tensor([0, 0, 1], dtype=ms.int32)  # Target indices
result = scatter(src, index, dim_size=2)
```

## Quick Start

### Basic Usage Flow

```python
from mindscience.e3nn import o3
import mindspore as ms

# 1. Define data types
irreps_in = o3.Irreps("3x0e + 2x1o")   # 3 scalars + 2 vectors
irreps_out = o3.Irreps("1x0e")         # 1 scalar output

# 2. Create equivariant layer
layer = o3.Linear(irreps_in, irreps_out)

# 3. Forward propagation
x = irreps_in.randn(-1)  # Generate input data
y = layer(x)             # Equivariant transformation

print(f"Input dimension: {x.shape}")
print(f"Output dimension: {y.shape}")
```

### Building Simple Networks

```python
from mindscience.e3nn import o3, nn
import mindspore as ms
import mindspore.nn as ms_nn
import mindspore.ops as ops

class SimpleE3NN(ms_nn.Cell):
    def __init__(self):
        super().__init__()
        # Feature extraction
        self.linear1 = o3.Linear("3x0e + 1x1o", "8x0e + 4x1o")
        self.act = nn.Activation("8x0e + 4x1o", acts=[ops.tanh, None])

        # Output layer
        self.linear2 = o3.Linear("8x0e + 4x1o", "1x0e")

    def construct(self, x):
        x = self.linear1(x)
        x = self.act(x)
        return self.linear2(x)

# Use the model
model = SimpleE3NN()
input_data = o3.Irreps("3x0e + 1x1o").randn(-1)
output = model(input_data)
```
