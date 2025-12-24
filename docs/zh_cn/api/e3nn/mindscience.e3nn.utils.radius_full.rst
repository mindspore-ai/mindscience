mindscience.e3nn.utils.radius_full
==================================

.. py:function:: mindscience.e3nn.utils.radius_full(x, y, batch_x=None, batch_y=None)

    为 `y` 中每个元素找到 `x` 中的所有点。

    参数：
        - **x** (Tensor) - x 节点特征矩阵。
        - **y** (Tensor) - y 节点特征矩阵。
        - **batch_x** (ndarray, 可选) - x 批向量。如果为 None，则根据 x 计算并返回。默认值：``None``。
        - **batch_y** (ndarray, 可选) - y 批向量。如果为 None，则根据 y 计算并返回。默认值：``None``。

    返回：
        - **edge_index** (numpy.ndarray) - 包括边的起点与终点。
        - **batch_x** (numpy.ndarray) - x 批向量。
        - **batch_y** (numpy.ndarray) - y 批向量。

    异常：
        - **ValueError** - 如果 `x` 和 `y` 的最后一个维度不匹配。