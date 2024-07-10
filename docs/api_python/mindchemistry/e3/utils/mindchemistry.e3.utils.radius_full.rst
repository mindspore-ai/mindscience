mindchemistry.e3.utils.radius_full
==================================

.. py:function:: mindchemistry.e3.utils.radius_full(x, y, batch_x=None, batch_y=None)

    找到 `x` 中每个元素在 `y` 中的所有点。

    参数：
        - **x** (Tensor) - 节点特征矩阵。
        - **y** (Tensor) - 节点特征矩阵。
        - **batch_x** (ndarray) - 批向量。默认值：``None``。
        - **batch_y** (ndarray) - 批向量。默认值：``None``。

    返回：
        - **edge_index** (numpy.ndarray) - 包括边的起点与终点。
        - **batch_x** (numpy.ndarray) - 批向量。
        - **batch_y** (numpy.ndarray) - 批向量。

    异常：
        - **ValueError** - 如果 `x` 和 `y` 的最后一个维度不匹配。