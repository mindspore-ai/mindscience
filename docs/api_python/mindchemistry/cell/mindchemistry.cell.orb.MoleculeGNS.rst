mindchemistry.cell.orb.MoleculeGNS
===================================

.. py:class:: mindchemistry.cell.orb.MoleculeGNS(num_node_in_features: int, num_node_out_features: int, num_edge_in_features: int, latent_dim: int, num_message_passing_steps: int, num_mlp_layers: int, mlp_hidden_dim: int, node_feature_names: List[str], edge_feature_names: List[str], use_embedding: bool = True, interactions: str = "simple_attention", interaction_params: Optional[Dict[str, Any]] = None)

    分子图神经网络。实现用于分子性质预测的灵活模块化图神经网络，基于注意力或其他交互机制的消息传递。支持节点和边嵌入、多个消息传递步骤，以及用于复杂分子图的可定制交互层。

    参数：
        - **num_node_in_features** (int) - 每个节点的输入特征数量。
        - **num_node_out_features** (int) - 每个节点的输出特征数量。
        - **num_edge_in_features** (int) - 每条边的输入特征数量。
        - **latent_dim** (int) - 节点和边表示的潜在维度。
        - **num_message_passing_steps** (int) - 消息传递层的数量。
        - **num_mlp_layers** (int) - 节点和边更新MLP的隐藏层数量。
        - **mlp_hidden_dim** (int) - MLP的隐藏维度大小。
        - **node_feature_names** (List[str]) - 从输入字典中使用的节点特征键列表。
        - **edge_feature_names** (List[str]) - 从输入字典中使用的边特征键列表。
        - **use_embedding** (bool，可选) - 是否对节点使用原子序数嵌入。默认值： ``True``。
        - **interactions** (str，可选) - 要使用的交互层类型（例如， ``"simple_attention"``）。默认值： ``"simple_attention"``。
        - **interaction_params** (Optional[Dict[str, Any]]，可选) - 交互层的参数，例如截断、多项式阶数、门类型。默认值： ``None``。

    输入：
        - **edge_features** (dict) - 边特征字典，必须包含 `edge_feature_names` 中指定的键。
        - **node_features** (dict) - 节点特征字典，必须包含 `node_feature_names` 中指定的键。
        - **senders** (Tensor) - 每条边的发送节点索引，形状为 :math:`(n_{edges},)`。
        - **receivers** (Tensor) - 每条边的接收节点索引，形状为 :math:`(n_{edges},)`。

    输出：
        - **edges** (dict) - 更新的边特征字典，键"feat"的形状为 :math:`(n_{edges}, latent\_dim)`。
        - **nodes** (dict) - 更新的节点特征字典，键"feat"的形状为 :math:`(n_{nodes}, latent\_dim)`。

    异常：
        - **ValueError** - 如果 `edge_features` 或 `node_features` 中缺少必需的特征键。
        - **ValueError** - 如果 `interactions` 不是支持的类型。