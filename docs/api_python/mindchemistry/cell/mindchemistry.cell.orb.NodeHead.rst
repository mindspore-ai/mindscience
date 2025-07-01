mindchemistry.cell.orb.NodeHead
===============================

.. py:class:: mindchemistry.cell.orb.NodeHead(latent_dim: int, num_mlp_layers: int, mlp_hidden_dim: int, target_property_dim: int, dropout: Optional[float] = None, remove_mean: bool = True)

    节点级预测头。

    实现用于从节点特征预测节点级属性的神经网络头。该头可以添加到基础模型中以在预训练期间启用辅助任务，或在微调步骤中添加。

    参数：
        - **latent_dim** (int) - 每个节点的输入特征维度。
        - **num_mlp_layers** (int) - MLP中的隐藏层数量。
        - **mlp_hidden_dim** (int) - MLP的隐藏维度大小。
        - **target_property_dim** (int) - 节点级目标属性的输出维度。
        - **dropout** (Optional[float]，可选) - MLP的dropout率。默认值： ``None``。
        - **remove_mean** (bool，可选) - 如果为True，从输出中移除均值，通常用于力预测。默认值： ``True``。

    输入：
        - **node_features** (dict) - 节点特征字典，必须包含键 "feat"，形状为 :math:`(n_{nodes}, latent\_dim)`。
        - **n_node** (Tensor) - 图中节点数量，形状为 :math:`(1,)`。

    输出：
        - **output** (dict) - 包含键 "node_pred" 的字典，值的形状为 :math:`(n_{nodes}, target\_property\_dim)`。

    异常：
        - **ValueError** - 如果 `node_features` 中缺少必需的特征键。