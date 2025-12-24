# DFT

## DFT Introduction

DFT (Discrete Fourier Transform) is a classic operation that maps discrete time-domain signals to the frequency domain, derived from the discretization of the continuous Fourier transform. Initially applied to signal analysis, it has now become a core tensor tool in deep learning and scientific computing. This module implements differentiable discrete transforms based on **DFT matrix (dense matrix) and matrix multiplication (matmul)**, including:

- Real Discrete Fourier Transform: `RDFTn` / `IRDFTn`
- Complex Discrete Fourier Transform: `DFTn` / `IDFTn`
- 1D Discrete Cosine Transform: `DCT` / `IDCT`
- 1D Discrete Sine Transform: `DST` / `IDST`

> Note: The interface design of this module is aligned as closely as possible with `scipy.fft` (refer to the [SciPy Documentation](https://docs.scipy.org/doc/scipy/reference/fft.html)), but does not fully cover all functions and parameters of `scipy.fft`. It is recommended to refer to the SciPy documentation for supplementary formulas when using it in comparison.

## Basic Formulas

- **Complex DFT (length N):**

   X[k] = Σ_{n=0}^{N-1} x[n] · e^{-i·2π·k·n/N}

- **Inverse DFT:**

  x[n] = (1/N) · Σ_{k=0}^{N-1} X[k] · e^{+i·2π·k·n/N}

- **Real RDFT and IRDFT:**

  For real-valued inputs, the frequency spectrum satisfies Hermitian symmetry, so only the **first half of the frequency components** (usually including the Nyquist component) need to be stored/calculated, thereby saving storage and computation resources. For additional support, please refer to the scipy.fft documentation.

## DFT Algorithm Flow

1. Parse input parameters: `shape`, `dim`, `norm`, `modes`, etc., and perform legality verification and standardization.
2. Construct a 1D DFT matrix according to the target `shape` during the initialization phase (to avoid repeated generation in each call).
3. Move the axes to be transformed to the last position to facilitate the use of matrix multiplication (`matmul`).
4. Perform 1D transform via matrix multiplication; for multi-dimensional cases, execute the operation axis by axis repeatedly to obtain n-dimensional results.
5. Restore the axis order, splice the output, and return the final tensor.

## Optimization Points of the `sciops.DFT` Module

The DFT operator in the sciops module has the following optimizations:

- **Separation of initialization and computation**: Complete DFT matrix generation and parameter verification during the operator initialization phase to avoid repeated construction and additional overhead at runtime, improving operational efficiency and interface stability.
- **Real input optimization (RDFTn / IRDFTn)**: For real inputs, only the first half of the frequency components are generated/stored (utilizing Hermitian symmetry), reducing the computation and storage volume by approximately half (depending on the specific value of N), and correctly reconstructing the complete spectrum during inverse transform.
- **Separate processing of real/imaginary parts to avoid complex types**: For scenarios where certain backend devices or numerical frameworks have poor support for complex numbers, the module provides an implementation path that separately calculates and stores real and imaginary parts to improve compatibility and numerical stability.
- **modes parameter support (additional feature)**: `DFTn`, `IDFTn`, `RDFTn`, `IRDFTn` provide the `modes` parameter to truncate output modes (default `None` means no truncation). When only low-frequency/partial modes are required, `modes` can be set to reduce matrix scale and computational cost, which is suitable for spectral truncation or reduced-order approximation scenarios.
- **Environment adaptation and compatibility implementation**: To avoid compatibility issues of native operators in certain backends (e.g., Ascend) and runtime modes, the module provides custom implementations of `MyRoll` and `MyFlip`, and dynamically selects the implementation based on the device runtime.

## Input/Output Size Description

- **Meaning of the `shape` parameter**: Generally, `shape` represents the 1D length of each axis to be transformed. For example, if the input tensor `x` has a shape of `(B, ..., M, N)` and you want to perform 2D DFT on the last two axes, you can set `shape=(M, N)`; you can also pass only a single axis length for 1D transform.

- **Input and output sizes of RDFTn / IRDFTn**:
  Let the trailing dimensions of the input tensor be: [..., n1, n2, ..., nk]
  where the last k dimensions need to undergo RDFTn (k = len(shape))
  For the last dimension:
  Input length: `N`
  Output length: `N//2 + 1`
  Output form of RDFTn: [..., n1, n2, ..., n_{k-1}, n_k//2 + 1]
  The input of IRDFTn must be: [..., n1, n2, ..., n_{k-1}, n_k//2 + 1]
  The output will restore the complete n_k: [..., n1, n2, ..., n_k]
- **Input and output sizes of DFTn / IDFTn**:
  For complex DFT, if full DFT is performed on an axis of length `N`, the output length remains `N`.
  When performing n-dimensional transform on multiple axes, `shape` should list the 1D length corresponding to each transformed axis, or specify the axis indices to be transformed during operator initialization.

## Detailed Description of the modes Parameter

This module additionally provides the `modes` parameter to truncate frequency-domain modes in certain dimensions, reducing computational cost.

For example: `modes=(None, 64)`

Indicates:

- No truncation for the previous dimension
- Only the first 64 frequencies are retained for the last dimension

Constraints on modes:

- Must be `≤ shape[i]//2`
- Default `None` means using full modes
- For the last dimension of RDFTn/IRDFTn, if `mode=None`, the standard `n//2+1` is used

This feature is **not supported by scipy.fft** and is an enhanced feature of this library, especially suitable for frequency cropping in PINN (Physics-Informed Neural Networks) / fluid modeling to reduce overhead.

## Comparison Examples with scipy.fft

The following are several minimal examples showing the corresponding usage of the module operators and `scipy.fft` for user reference:

- **RDFT vs scipy.fft.rfft**

```python
# Using sciops
rdft = RDFTn(shape=(N,))
out_real, out_imag = rdft(x_real)  

# Corresponding scipy implementation
from scipy.fft import rfft
X = rfft(x_real, n=N)

```

-**DFT vs scipy.fft.fft**

```python
# Using sciops
dft = DFTn(shape=(N,))
Xr, Xi = dft(x_complex_as_two_arrays)

# Corresponding scipy implementation
from scipy.fft import fft
X = fft(x_complex, n=N)

```

(Please adjust the examples according to the actual return format of the module: if the operator returns separated real/imaginary parts, the examples should be consistent.)

## Usage Examples

- RDFT Usage Example

```python

import mindspore as ms
from mindscience.sciops import RDFTn

x = ms.ops.rand((2, 32, 512))
rdft = RDFTn(shape=x.shape[-2:])
br, bi = rdft(x)

```

- DCT Usage Example

```python
import mindspore as ms
from mindscience.sciops import DCT

a = ms.ops.rand((4, 128))
dct = DCT(shape=(128,))
b = dct(a)

```
