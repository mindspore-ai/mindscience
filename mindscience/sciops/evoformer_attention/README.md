简体中文 | [English](README_EN.md)

# EvoformerAttention

## EvoformerAttention 介绍

EvoformerAttention 是一种面向科学计算的高效注意力机制，专为处理蛋白质结构预测等生物大模型中的复杂依赖关系而设计。它源自 AlphaFold2 中的 Evoformer 模块，支持行注意力、列注意力和三角形注意力等多种非标准注意力模式，能够有效建模多序列比对（MSA）和残基对之间的长程相互作用。与传统 Transformer 中的自注意力不同，EvoformerAttention 在计算过程中引入了额外的偏置项和掩码结构，以适应生物学先验知识。

相比于通用注意力实现，EvoformerAttention 具有以下优点：

- **领域适配性强**：针对结构生物学任务定制，天然支持 MSA 和残基对表示；
- **内存效率高**：通过融合计算与显存优化策略，显著降低峰值显存占用，使大规模模型训练成为可能。

## 使用样例

- 基础矩阵注意力计算：

```python
import numpy as np
import mindspore as ms
from mindspore import Tensor
from mindscience.sciops import evo_attention

b, n, s, d = 2048, 1, 2048, 8
scale_value = 1.0 / np.sqrt(d)

query = Tensor(np.random.uniform(-0.1, 0.1, (b, s, n, d)), ms.bfloat16)
key = Tensor(np.random.uniform(-0.1, 0.1, (b, s, n, d)), ms.bfloat16)
value = Tensor(np.random.uniform(-0.1, 0.1, (b, s, n, d)), ms.bfloat16)
bias = Tensor(np.random.uniform(-0.1, 0.1, (1, n, s, s)), ms.bfloat16)

mask = np.concatenate([np.ones((b, 1, 1, -5)),
                       np.zeros((b, 1, 1, 5))], axis=-1)
evo_mask = Tensor(1 - mask.astype(np.uint8))

output = evo_attention(query, key, value, n, bias, evo_mask,scale_value, input_layout="BSND")
```

- 与标准注意力计算对比验证：

```python
import numpy as np
import mindspore as ms
from mindspore import Tensor
from mindscience.sciops import evo_attention

def standard_attention(q, k, v, mask, scale):
    logits = ms.ops.BatchMatMul(transpose_b=True)(q, k) * scale
    logits = logits + mask.astype(ms.float16) * (-1e12)  
    attn_weights = ms.ops.Softmax()(logits)
    return ms.ops.BatchMatMul()(attn_weights, v)

standard_mask = Tensor((mask - 1) * 1e12, ms.float16)
standard_output = standard_attention(query, key, value, standard_mask, scale_value)

ms.ops.allclose(output, standard_output, atol=1e-4, rtol=1e-7)
```