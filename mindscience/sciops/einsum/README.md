简体中文 | [English](README_EN.md)

# Einsum

## Einsum介绍

Einsum（Einstein summation）是一种强大而灵活的张量运算系统，最初由爱因斯坦提出用于相对论计算。在现代深度学习和科学计算中，它已经成为处理多维张量运算的重要工具。Einsum的核心思想是通过标记张量的维度来指定运算规则，例如，`Einsum("ij,jk->ik", a, b)`表示矩阵乘法，其中i、j和k分别标记了维度，箭头后面的ik表示输出张量的维度。这种表示方法不仅可以实现矩阵乘法，还能表达转置、求迹、内积等多种运算。

相比于传统的张量运算方法，Einsum具有以下优点：

- 语法简洁，通过一行代码就能够表达复杂的张量运算；
- 适用范围广，从简单的张量运算到复杂的深度学习模型都能应用。

## Einsum算法流程

Einsum算子的算法主要流程如下：

1. 解析输入表达式，并计算出计算量最小的规约路径；
2. 规约路径确定后，进行标签排布优化；
3. 基于优化后的规约路径和标签排布，构建每一步计算步骤的信息并保存；
4. 将输入的张量压栈，每次从栈顶取出两个张量，根据对应的计算步骤进行计算，再将计算结果重新入栈；
5. 重复执行步骤4中的出栈、计算、入栈过程，直到栈中只有一个张量即为最终结果。

## sciops.Einsum优化点

sciops模块中的Einsum算子进行如下的优化：

- 计算步骤优化：省去了不必要的中间计算过程，简化了计算步骤，而且把解析相关的一次性计算放在算子初始化中，减少了不必要的重复计算；
- 标签排布优化：Einsum算法中依赖统一的标签排布，即标签间的相对顺序。不同的标签排布对应的transpose次数不同，从而影响最终的算子性能。sciops.Einsum实现了标签排布和transpose数量的关系建模，从而得到transpose数量较少的标签排布。
- matmul与transpose融合：MindSpore框架中的ops.BatchMatMul提供了transpose_a和transpose_b参数，使得matmul和transpose能够进行融合计算，减少了片上内存的访存开销。

## 使用样例

- 矩阵转置运算：

```python
import numpy as np

import mindspore as ms
from mindscience.sciops import Einsum

einsum = Einsum("ij->ji", use_opt=True)
x = ms.Tensor(np.random.randn(4, 5), ms.float32)
y = einsum(x)
```

- 矩阵规约运算：

```python
import numpy as np

import mindspore as ms
from mindscience.sciops import Einsum

einsum = Einsum("ij->i", use_opt=True)
x = ms.Tensor(np.random.randn(4, 5), ms.float32)
y = einsum(x)
```

- 矩阵乘运算：

```python
import numpy as np

import mindspore as ms
from mindscience.sciops import Einsum

einsum = Einsum("ij,jk->ik", use_opt=True)
x = ms.Tensor(np.random.randn(3, 4), ms.float32)
y = ms.Tensor(np.random.randn(4, 5), ms.float32)
z = einsum(x, y)
```

更多使用样例可参考[Einsum测试用例](../../../tests/sciops/test_einsum.py)。
