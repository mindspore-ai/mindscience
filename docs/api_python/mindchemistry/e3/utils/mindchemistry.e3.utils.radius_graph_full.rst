mindchemistry.e3.utils.radius_graph_full
=========================================

.. py:function:: mindchemistry.e3.utils.radius_graph_full(x, batch=None, loop=False, flow='source_to_target')

    计算给定距离内图所有点之间的边。

    参数：
        - **x** (Tensor) - 节点特征矩阵。
        - **batch** (Tensor) - 批向量。如果为 None，则计算并返回。默认值：``None``。
        - **loop** (bool) - 图中是否包含自环。默认值：``False``。
        - **flow** (str) - {'source_to_target', 'target_to_source'}，与消息传递结合使用时的流向。默认值：``'source_to_target'``。

    返回：
        - **edge_index** (ndarray) - 包括边的起点与终点。
        - **batch** (ndarray) - 批向量。

    异常：
        - **ValueError** - 如果 `flow` 不在 {'source_to_target', 'target_to_source'} 中。