[简体中文](README.md) | English

# EvoformerAttention

## Introduction

EvoformerAttention is an efficient attention mechanism for scientific computing, specifically designed to handle complex dependencies in biological large models such as protein structure prediction. Derived from the Evoformer module in AlphaFold2, it supports multiple non-standard attention patterns including row attention, column attention, and triangular attention, enabling effective modeling of long-range interactions between Multiple Sequence Alignments (MSA) and residue pairs. Unlike self-attention in traditional Transformers, EvoformerAttention incorporates additional bias terms and mask structures during computation to adapt to biological prior knowledge.

Compared with general-purpose attention implementations, EvoformerAttention offers the following advantages:

- **Strong domain adaptability**: Customized for structural biology tasks, it natively supports MSA and residue pair representations.
- **High memory efficiency**: Through fused computation and memory optimization strategies, it significantly reduces peak memory usage, enabling large-scale model training.

## Using Cases

- matrix attention computation：

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

- comparison with standard attention computation：

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