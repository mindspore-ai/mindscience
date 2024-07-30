mindchemistry.e3.utils.Ncon
============================

.. py:class:: mindchemistry.e3.utils.Ncon(irreps_in, act, normalize=True, epsilon=None, bias=False, init_method='zeros', dtype=float32, ncon_dtype=float32)

    多个张量的缩并运算符，功能类似于 Einsum。

    参数：
        - **con_list** (List[List[int]]) - 每个张量的索引列表。每个列表中的数目应与相应张量的维度相对应。正索引表示要缩并或求和的维度。负索引表示要保留的维度（作为批维度）。

    输入：
        - **inputs** (List[Tensor]) - 张量列表。

    输出：
        - **latents** (Tensor) - 结果张量，形状取决于输入和运算过程。

    异常：
        - **ValueError**: 如果命令的数量与操作的数量不匹配。
