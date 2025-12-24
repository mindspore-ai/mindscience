简体中文 | [English](README_EN.md)

# FFT

## FFT介绍

FFT（Fast Fourier Transform，快速傅里叶变换）是离散傅里叶变换（DFT）的高效实现，通过分治策略将计算复杂度从O(N²)降至O(NlogN)，核心用于时域与频域信号转换，逆变换（IFFT）实现频域还原。实值场景下的RFFT/IRFFT利用共轭对称性，减少计算量与存储开销。

Mindscience.sciops的asd_fft系列算子适配Ascend硬件平台，兼容MindSpore框架，优势如下：

- 高精度，与 NumPy 结果误差≤1e-3，满足科学计算需求；
- 高性能，拆解高维 FFT，融合转置与矩阵乘，降低访存延迟；
- 易使用，支持多维度、形状扩展配置，兼容动态图模式；
- 梯度友好，优化 RFFT/IRFFT 梯度计算，保证精度一致。

## 使用样例

- 复-复 FFT/IFFT：

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

- 实-复 RFFT：

```python
import numpy as np
import mindspore as ms
from mindscience.sciops import asd_rfftn

ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE)
ar = ms.Tensor(np.random.rand(2, 16, 16), ms.float32)
br, bi = asd_rfftn(ar, ndim=2)
```

- 复-实 IRFFT：

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

- FFT 梯度计算：

```python
import numpy as np
import mindspore as ms
from mindspore import ops, value_and_grad
from mindscience.sciops import asd_fftn

ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE)
def loss_func(yr, yi): return ops.sum(yr**2 + 2*yi**2)
def forward_fn(xr, xi): return loss_func(*asd_fftn(xr, xi, ndim=2))

ar = ms.Tensor(np.random.rand(2, 16, 16), ms.float32)
ai = ms.Tensor(np.random.rand(2, 16, 16), ms.float32)
grad_fn = ms.value_and_grad(forward_fn, grad_position=(0, 1))
output, (ar_grad, ai_grad) = grad_fn(ar, ai)
```

更多使用样例可参考[FFT测试用例](../../../tests/sciops/test_asd_fft.py)。
