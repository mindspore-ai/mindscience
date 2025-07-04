mindchemistry.cell.orb.AttentionInteractionNetwork
==================================================

.. py:class:: mindchemistry.cell.orb.AttentionInteractionNetwork(num_node_in: int, num_node_out: int, num_edge_in: int, num_edge_out: int, num_mlp_layers: int, mlp_hidden_dim: int, attention_gate: str = "sigmoid", distance_cutoff: bool = True, polynomial_order: int = 4, cutoff_rmax: float = 6.0)

    注意力交互网络。实现基于注意力机制的消息传递神经网络层，用于分子图的边更新。

    参数：
        - **num_node_in** (int) - 节点输入特征数量。
        - **num_node_out** (int) - 节点输出特征数量。
        - **num_edge_in** (int) - 边输入特征数量。
        - **num_edge_out** (int) - 边输出特征数量。
        - **num_mlp_layers** (int) - 节点和边更新MLP的隐藏层数量。
        - **mlp_hidden_dim** (int) - MLP的隐藏维度大小。
        - **attention_gate** (str，可选) - 注意力门类型， ``"sigmoid"`` 或 ``"softmax"``。默认值： ``"sigmoid"``。
        - **distance_cutoff** (bool，可选) - 是否使用基于距离的边截断。默认值： ``True``。
        - **polynomial_order** (int，可选) - 多项式截断函数的阶数。默认值： ``4``。
        - **cutoff_rmax** (float，可选) - 截断的最大距离。默认值： ``6.0``。

    输入：
        - **graph_edges** (dict) - 边特征字典，必须包含键"feat"，形状为 :math:`(n_{edges}, num\_edge\_in)`。
        - **graph_nodes** (dict) - 节点特征字典，必须包含键"feat"，形状为 :math:`(n_{nodes}, num\_node\_in)`。
        - **senders** (Tensor) - 每条边的发送节点索引，形状为 :math:`(n_{edges},)`。
        - **receivers** (Tensor) - 每条边的接收节点索引，形状为 :math:`(n_{edges},)`。

    输出：
        - **edges** (dict) - 更新的边特征字典，键"feat"的形状为 :math:`(n_{edges}, num\_edge\_out)`。
        - **nodes** (dict) - 更新的节点特征字典，键"feat"的形状为 :math:`(n_{nodes}, num\_node\_out)`。

    异常：
        - **ValueError** - 如果 `attention_gate` 不是"sigmoid"或"softmax"。
        - **ValueError** - 如果边或节点特征不包含必需的"feat"键。