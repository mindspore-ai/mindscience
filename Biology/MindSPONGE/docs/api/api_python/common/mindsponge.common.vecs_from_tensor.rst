mindsponge.common.vecs_from_tensor
==================================

.. py:function:: mindsponge.common.vecs_from_tensor(inputs)

    将输入表示位置信息的tensor在最后一根轴拆分，化为向量， `vecs_to_tensor` 的逆操作。

    参数：
        - **inputs** (Tensor) - 原子位置信息，shape为 :math:`(..., 3)`。

    返回：
        返回带有三个tensor的tuple :math:`(x, y, z)`，分别包含x, y, z坐标信息。