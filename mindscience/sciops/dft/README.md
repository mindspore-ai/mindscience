# DFT

## DFT 介绍

DFT（离散傅里叶变换）是将离散时域信号映射到频域的经典运算，由连续傅里叶变换离散化而来。最初用于信号分析，现已是深度学习与科学计算的核心张量工具。此模块实现基于 **DFT 矩阵(dense matrix)和 矩阵乘法（matmul）** 的可微离散变换，包含：

- 实数离散傅里叶变换：`RDFTn` / `IRDFTn`
- 复数离散傅里叶变换：`DFTn` / `IDFTn`
- 一维离散余弦变换：`DCT` / `IDCT`
- 一维离散正弦变换：`DST` / `IDST`

> 说明：本模块的接口设计与 `scipy.fft` 保持尽量对齐（可参考[SciPy 文档](https://docs.scipy.org/doc/scipy/reference/fft.html))，但并未完全覆盖 `scipy.fft` 的所有函数与参数。建议在对照使用时同时参考 SciPy 的文档以补充公式。

## 基本公式

- **复数 DFT（长度 N）：**

  X[k] = Σ_{n=0}^{N-1} x[n] · e^{-i·2π·k·n/N}

- **逆 DFT：**

  x[n] = (1/N) · Σ_{k=0}^{N-1} X[k] · e^{+i·2π·k·n/N}

- **实数 RDFT 与 IRDFT：**

  对实值输入，频谱满足共轭对称性(Hermitian symmetry)，因此仅需存储/计算 **前半频率分量**（通常含 Nyquist 分量），从而节省存储与计算。如需其他支持，请参照 scipy.fft 文档。

## DFT 算法流程

1. 解析输入参数：`shape`、`dim`、`norm`、`modes` 等，进行合法性校验与标准化。
2. 在初始化阶段根据目标 `shape` 构造一维 DFT 矩阵（避免每次调用重复生成）。
3. 将待变换轴移动到最后以便使用矩阵乘法（`matmul`）。
4. 通过矩阵乘法执行一维变换；对多维情况逐轴重复执行以得到 n 维结果。
5. 恢复轴顺序并拼接输出，返回最终张量。

## `sciops.DFT`  模块优化点

sciops 模块中的 DFT 算子进行了如下优化：

- **初始化与计算分离**：在算子初始化阶段完成 DFT 矩阵生成与参数校验，避免运行时重复构建与额外开销，提高运行效率和接口稳定性。
- **实数输入优化（RDFTn / IRDFTn）**：针对实数输入只生成/存储前半频率分量（利用 Hermitian 对称性），将计算与存储量约减一半（视具体 N 而定），并在逆变换时正确重构完整频谱。
- **实部 / 虚部分别处理以规避复数类型**：对某些后端设备或数值框架对复数不友好的情况，模块提供将实部与虚部分开计算/存储的实现路径，以提高兼容性和数值稳定性。
- **modes 参数支持（额外特性）**：`DFTn`、`IDFTn`、`RDFTn`、`IRDFTn` 提供 `modes` 参数用于截断输出模态（默认 `None` 表示不截断）。当只需低频/部分模式时可设置 `modes`，从而减少矩阵规模与计算量，适用于谱截断或降阶近似场景。
- **环境适配与兼容实现**：为规避某些后端（如 Ascend）和运行模式下原生算子的兼容问题，模块提供自定义 `MyRoll`、`MyFlip` 实现，并根据设备运行时动态选择实现。

## 输入/输出 尺寸说明

- **`shape` 参数的含义**：通常 `shape` 表示对每个要变换维度的一维长度。例如当输入张量 `x` 形状为 `(B, ..., M, N)`，若希望对最后两个轴做二维 DFT，可设置 `shape=(M, N)`；也可仅传单轴长度来做 1D 变换。

- **RDFTn / IRDFTn 的输入输出尺寸**：
  设输入张量的末尾维度为：[..., n1, n2, ..., nk]
  其中最后 k 个维度需要做 RDFTn（k = len(shape))
  对于最后一个维度:
  输入长度：`N`
  输出长度：`N//2 + 1`
  RDFTn的输出形式：[..., n1, n2, ..., n_{k-1}, n_k//2 + 1]
  IRDFTn的输入必须是：[..., n1, n2, ..., n_{k-1}, n_k//2 + 1]
  输出将恢复为完整的 n_k：[..., n1, n2, ..., n_k]
- **DFTn / IDFTn 的输入输出尺寸**:
  对于复数 DFT，若对长度为 `N` 的轴做全量 DFT，则输出长度依然为 `N`。
  当对多轴做 n 维变换时，`shape` 应列出每个变换轴对应的一维长度，或在算子初始化时指定要变换的轴索引。

## modes 参数的详细说明

本模块额外提供 `modes` 参数，用于在某些维度上 截断频域模态，减少计算量。

例如：`modes = (None, 64)`

表示：

- 前一维不截断
- 最后一维只保留前 64 个频率

modes 的约束：

- 必须 `≤ shape[i]//2`
- 默认 `None` 表示使用全量模态
- 对 RDFTn/IRDFTn 最后一维若 `mode=None` 使用标准 `n//2+1`

该功能 **scipy.fft 并不支持**，是本库强化特性，特别适用于 PINN / 流体建模中频率裁剪以降低开销。

## 与 scipy.fft 的对比示例

下面给出若干最小示例，展示模块算子与 `scipy.fft` 的对应用法，供用户参考：

- **RDFT 与 scipy.fft.rfft**

```python
# 使用 sciops
rdft = RDFTn(shape=(N,))
out_real, out_imag = rdft(x_real)  

# 对应 scipy
from scipy.fft import rfft
X = rfft(x_real, n=N)

```

- **DFT 与 scipy.fft.fft**

```python
# 使用 sciops
dft = DFTn(shape=(N,))
Xr, Xi = dft(x_complex_as_two_arrays)

# 对应 scipy
from scipy.fft import fft
X = fft(x_complex, n=N)

```

（请根据模块实际返回格式调整示例：若算子返回分开的实部/虚部，示例中应保持一致。）

## 使用样例

- RDFT 使用样例

```python
import mindspore as ms
from mindscience.sciops import RDFTn

x = ms.ops.rand((2, 32, 512))
rdft = RDFTn(shape=x.shape[-2:])
br, bi = rdft(x)
```

- DCT 使用样例

```python
import mindspore as ms
from mindscience.sciops import DCT

a = ms.ops.rand((4, 128))
dct = DCT(shape=(128,))
b = dct(a)

```

