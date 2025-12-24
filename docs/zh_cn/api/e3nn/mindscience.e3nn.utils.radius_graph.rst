mindscience.e3nn.utils.radius_graph
=========================================

.. py:function:: mindscience.e3nn.utils.radius_graph(x, r, batch=None, loop=False, max_num_neighbors=32, flow='source_to_target')

    计算给定距离内图所有点之间的边。

    参数：
        - **x** (ndarray) - 节点特征矩阵。
        - **r** (Union[ndarray, float]) - 半径。
        - **batch** (ndarray, 可选) - 批向量。如果为 None，则计算并返回。默认值：``None``。
        - **loop** (bool, 可选) - 图中是否包含自环。默认值：``False``。
        - **max_num_neighbors** (int, 可选) - 返回每个 `y` 元素的最大邻居数量。默认值：``32``。
        - **flow** (str, 可选) - {'source_to_target', 'target_to_source'}，与消息传递结合使用时的流向。默认值：``'source_to_target'``。

    返回：
        - **edge_index** (numpy.ndarray) - 包括边的起点与终点。
        - **batch** (numpy.ndarray) - 批向量。

    异常：
        - **ValueError** - 如果 `flow` 不是 {'source_to_target', 'target_to_source'} 之一。