mindchemistry.e3.utils.radius
==============================

.. py:function:: mindchemistry.e3.utils.radius(x, y, r, batch_x=None, batch_y=None, max_num_neighbors=32)

    在 `x` 中找到每个 `y` 元素在距离 `r` 内的所有点。

    参数：
        - **x** (ndarray) - x 节点特征矩阵。
        - **y** (ndarray) - y 节点特征矩阵。
        - **r** (ndarray, float) - 半径。
        - **batch_x** (ndarray) - x 批向量。如果为 None，则根据 x 计算并返回。默认值：``None``。
        - **batch_y** (ndarray) - y 批向量。如果为 None，则根据 y 计算并返回。默认值：``None``。
        - **max_num_neighbors** (int) - 返回每个 `y` 元素的最大邻居数量。默认值：``32``。

    返回：
        - **edge_index** (numpy.ndarray) - 包括边的起点与终点。
        - **batch_x** (numpy.ndarray) - x 批向量。
        - **batch_y** (numpy.ndarray) - y 批向量。

    异常：
        - **ValueError** - 如果 `x` 和 `y` 的最后一个维度不匹配。