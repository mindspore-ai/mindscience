mindchemistry.cell.orb.GraphHead
=================================

.. py:class:: mindchemistry.cell.orb.GraphHead(latent_dim: int, num_mlp_layers: int, mlp_hidden_dim: int, target_property_dim: int, node_aggregation: str = "mean", dropout: Optional[float] = None, compute_stress: Optional[bool] = False)

    图级预测头。实现可以附加到基础模型的图级预测头，用于从节点特征预测图级属性（例如应力张量），使用聚合和MLP。

    参数：
        - **latent_dim** (int) - 每个节点的输入特征维度。
        - **num_mlp_layers** (int) - MLP中的隐藏层数量。
        - **mlp_hidden_dim** (int) - MLP的隐藏维度大小。
        - **target_property_dim** (int) - 图级属性的输出维度。
        - **node_aggregation** (str，可选) - 节点预测的聚合方法，例如 ``"mean"`` 或 ``"sum"``。默认值： ``"mean"``。
        - **dropout** (Optional[float]，可选) - MLP的dropout率。默认值： ``None``。
        - **compute_stress** (bool，可选) - 是否计算和输出应力张量。默认值： ``False``。

    输入：
        - **node_features** (dict) - 节点特征字典，必须包含键"feat"，形状为 :math:`(n_{nodes}, latent\_dim)`。
        - **n_node** (Tensor) - 图中节点数量，形状为 :math:`(1,)`。

    输出：
        - **output** (dict) - 包含键"stress_pred"的字典，值的形状为 :math:`(1, target\_property\_dim)`。

    异常：
        - **ValueError** - 如果 `node_features` 中缺少必需的特征键。