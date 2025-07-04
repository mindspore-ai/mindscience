mindchemistry.cell.orb.EnergyHead
==================================

.. py:class:: mindchemistry.cell.orb.EnergyHead(latent_dim: int, num_mlp_layers: int, mlp_hidden_dim: int, target_property_dim: int, predict_atom_avg: bool = True, reference_energy_name: str = "mp-traj-d3", train_reference: bool = False, dropout: Optional[float] = None, node_aggregation: Optional[str] = None)

    图级能量预测头。实现用于预测分子图总能量或原子平均能量的神经网络头。支持节点级聚合、参考能量偏移和灵活的输出模式。

    参数：
        - **latent_dim** (int) - 每个节点的输入特征维度。
        - **num_mlp_layers** (int) - MLP中的隐藏层数量。
        - **mlp_hidden_dim** (int) - MLP的隐藏维度大小。
        - **target_property_dim** (int) - 能量属性的输出维度（通常为1）。
        - **predict_atom_avg** (bool，可选) - 是否预测每原子平均能量而不是总能量。默认值： ``True``。
        - **reference_energy_name** (str，可选) - 用于偏移的参考能量名称，例如 ``"vasp-shifted"``。默认值： ``"mp-traj-d3"``。
        - **train_reference** (bool，可选) - 是否将参考能量训练为可学习参数。默认值： ``False``。
        - **dropout** (Optional[float]，可选) - MLP的dropout率。默认值： ``None``。
        - **node_aggregation** (str，可选) - 节点预测的聚合方法，例如 ``"mean"``或 ``"sum"``。默认值： ``None``。

    输入：
        - **node_features** (dict) - 节点特征字典，必须包含键"feat"，形状为 :math:`(n_{nodes}, latent\_dim)`。
        - **n_node** (Tensor) - 图中节点数量，形状为 :math:`(1,)`。

    输出：
        - **output** (dict) - 包含键"graph_pred"的字典，值的形状为 :math:`(1, target\_property\_dim)`。

    异常：
        - **ValueError** - 如果 `node_features` 中缺少必需的特征键。
        - **ValueError** - 如果 `node_aggregation` 不是支持的类型。