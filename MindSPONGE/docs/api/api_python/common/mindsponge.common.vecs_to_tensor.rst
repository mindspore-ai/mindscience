mindsponge.common.vecs_to_tensor
================================

.. py:function:: mindsponge.common.vecs_to_tensor(v)

    将向量转化为最后一维度为3的tensor，和vecs_from_tensor操作相反。

    参数：
        - **v** (tuple) - 带有3个tensor，分别表示位置坐标中的x, y, z。

    返回：
        tensor，返回三个tensor在最后一根轴合并后结果，shape为 :math:`(..., 3)` 。
