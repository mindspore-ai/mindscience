[简体中文](README.md) | English

# FFT

## Introduction

FFT (Fast Fourier Transform) is an efficient implementation of the Discrete Fourier Transform (DFT). It reduces the computational complexity from O(N²) to O(NlogN) through a divide-and-conquer strategy, and is primarily used for converting signals between the time domain and frequency domain. Its inverse transform (IFFT) restores frequency-domain signals back to the time domain. In real-valued signal processing scenarios, RFFT (Real Fast Fourier Transform) and IRFFT (Inverse Real Fast Fourier Transform) leverage the conjugate symmetry of complex numbers to reduce computational and storage overhead.

The asd_fft series of operators (asd_fftn/asd_ifftn/asd_rfftn/asd_irfftn) in mindscience.sciops are optimized for the Ascend hardware platform and fully compatible with the MindSpore framework, offering the following key advantages:

- **High precision**:The error between calculation results and NumPy's standard implementation is ≤1e-3, meeting the stringent precision requirements of scientific computing.
- **High performance**: By decomposing high-dimensional FFT computation logic and fusing transpose and matrix multiplication operations, it significantly reduces memory access latency and improves computational efficiency.
- **Ease of use**: Supports multi-dimensional computation, flexible shape extension configurations, and is fully compatible with MindSpore's dynamic graph mode.
- **Gradient-friendly**: Special optimization for the gradient computation process of RFFT/IRFFT ensures consistent precision between forward computation and backward gradients.

## Using Cases

- Complex-to-Complex FFT/IFFT：

```python
import numpy as np
import mindspore as ms
from mindscience.sciops import asd_fftn, asd_ifftn

ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE)
shape = (2, 16, 16)
a = np.random.rand(*shape) + 1j * np.random.rand(*shape)
ar, ai = ms.Tensor(a.real, ms.float32), ms.Tensor(a.imag, ms.float32)

br, bi = asd_fftn(ar, ai, ndim=2)
ar_restored, ai_restored = asd_ifftn(br, bi, ndim=2)
```

- Real-to-Complex RFFT：

```python
import numpy as np
import mindspore as ms
from mindscience.sciops import asd_rfftn

ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE)
ar = ms.Tensor(np.random.rand(2, 16, 16), ms.float32)
br, bi = asd_rfftn(ar, ndim=2)
```

- Complex-to-Real IRFFT：

```python
import numpy as np
import mindspore as ms
from mindscience.sciops import asd_rfftn, asd_irfftn

ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE)
ar = ms.Tensor(np.random.rand(2, 16, 16), ms.float32)
br, bi = asd_rfftn(ar, ndim=2)

ar_restored = asd_irfftn(br, bi, ndim=2)
ar_extended = asd_irfftn(br, bi, n=ar.shape[-1]+1, ndim=2)
```

- FFT gradient：

```python
import numpy as np
import mindspore as ms
from mindspore import ops
from mindscience.sciops import asd_fftn

ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE)
def loss_func(yr, yi): return ops.sum(yr**2 + 2*yi**2)
def forward_fn(xr, xi): return loss_func(*asd_fftn(xr, xi, ndim=2))

ar = ms.Tensor(np.random.rand(2, 16, 16), ms.float32)
ai = ms.Tensor(np.random.rand(2, 16, 16), ms.float32)
grad_fn = ms.value_and_grad(forward_fn, grad_position=(0, 1))
output, (ar_grad, ai_grad) = grad_fn(ar, ai)
```

For more using cases, please refer to the [FFT test cases](../../../tests/sciops/test_asd_fft.py).
