[简体中文](README.md) | English

# Einsum

## Introduction

Einsum (Einstein summation) is a powerful and flexible tensor operation system originally proposed by Einstein for relativistic calculations. In modern deep learning and scientific computing, it has become an important tool for handling multidimensional tensor operations. The core idea of Einsum is to specify the operation rules by labeling the dimensions of tensors, for example, `Einsum("ij,jk->ik", a, b)` represents matmul, "i", "j", and "k" respectively mark the dimensions, and "ik" after the arrow represents the dimension of the output tensor. This representation method not only can implement matrix multiplication, but also can express various operations such as transpose, trace, inner product, etc.

Compared to traditional tensor operation methods, Einsum has the following advantages:

- The syntax is concise, and complex tensor operations can be expressed with just one line of code;
- It has a wide range of applications, from simple tensor operations to complex deep learning models.

## Algorithm Procedure

The procedure of the Einsum is as follows:

1. Analyze the input expression and calculate the least computationally reduction path;
2. After determining the reduction path, optimize the label layout;
3. Based on the optimized reduction path and label layout, construct information for each calculation step and save it;
4. Push the input tensor into stack, pop two tensors from the top of the stack each time, calculate according to the corresponding calculation steps, and then push the calculation results into stack again;
5. Repeat the process of popping, calculating, and pushing in step 4 until there is only one tensor in the stack, which is the final result.

## Optimization

The optimization of the Einsum in the sciops module are as follows:

- Optimization of calculation steps: unnecessary intermediate calculation processes are eliminated, simplifying calculation steps, and one-time calculations related to parsing are placed in operator initialization, reducing unnecessary repetitive calculations;
- Optimization of label layout: Einsum algorithm relies on a unified label layout, that is, the relative order between labels. Different label arrangements correspond to different transpose times, which affects the final operator performance. Einsum in sciops has implemented modeling of the relationship between label arrangement and the number of transposes, resulting in label arrangements with fewer transposes.
- Matmul and transpose fusion: ops.BatchMatMul in the MindSpore framework provides "transpose_a" and "transpose_b" arguments, enabling Matmul and transpose to perform fusion calculations and reducing on-chip memory access overhead.

## Using Cases

- transpose：

```python
import numpy as np

import mindspore as ms
from mindscience.sciops import Einsum

einsum = Einsum("ij->ji", use_opt=True)
x = ms.Tensor(np.random.randn(4, 5), ms.float32)
y = einsum(x)
```

- reduction：

```python
import numpy as np

import mindspore as ms
from mindscience.sciops import Einsum

einsum = Einsum("ij->i", use_opt=True)
x = ms.Tensor(np.random.randn(4, 5), ms.float32)
y = einsum(x)
```

- matmul：

```python
import numpy as np

import mindspore as ms
from mindscience.sciops import Einsum

einsum = Einsum("ij,jk->ik", use_opt=True)
x = ms.Tensor(np.random.randn(3, 4), ms.float32)
y = ms.Tensor(np.random.randn(4, 5), ms.float32)
z = einsum(x, y)
```

For more using cases, please refer to the [Einsum test cases](../../../tests/sciops/test_einsum.py).
