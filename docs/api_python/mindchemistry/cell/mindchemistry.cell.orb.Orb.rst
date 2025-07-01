mindchemistry.cell.orb.Orb
===========================

.. py:class:: mindchemistry.cell.orb.Orb(model: MoleculeGNS, node_head: Optional[NodeHead] = None, graph_head: Optional[GraphHead] = None, stress_head: Optional[GraphHead] = None, model_requires_grad: bool = True, cutoff_layers: Optional[int] = None)

    Orb图回归器。将预训练的基础模型（如MoleculeGNS）与可选的节点、图和应力回归头结合，支持微调或特征提取工作流程。

    参数：
        - **model** (MoleculeGNS) - 用于消息传递和特征提取的预训练或随机初始化基础模型。
        - **node_head** (NodeHead，可选) - 节点级属性预测的回归头。默认值： ``None``。
        - **graph_head** (GraphHead，可选) - 图级属性预测（例如能量）的回归头。默认值： ``None``。
        - **stress_head** (GraphHead，可选) - 应力预测的回归头。默认值： ``None``。
        - **model_requires_grad** (bool，可选) - 是否微调基础模型（True）或冻结其参数（False）。默认值： ``True``。
        - **cutoff_layers** (int，可选) - 如果提供，仅使用基础模型的前 ``"cutoff_layers"`` 个消息传递层。默认值： ``None``。

    输入：
        - **edge_features** (dict) - 边特征字典（例如，`{"vectors": Tensor, "r": Tensor}`）。
        - **node_features** (dict) - 节点特征字典（例如，`{"atomic_numbers": Tensor, ...}`）。
        - **senders** (Tensor) - 每条边的发送节点索引。形状：:math:`(n_{edges},)`。
        - **receivers** (Tensor) - 每条边的接收节点索引。形状：:math:`(n_{edges},)`。
        - **n_node** (Tensor) - 批次中每个图的节点数量。形状：:math:`(n_{graphs},)`。

    输出：
        - **output** (dict) - 包含以下内容的字典：
        
          - **edges** (dict) - 消息传递后的边特征，例如 `{..., "feat": Tensor}`。
          - **nodes** (dict) - 消息传递后的节点特征，例如 `{..., "feat": Tensor}`。
          - **graph_pred** (Tensor) - 图级预测，例如能量。形状：:math:`(n_{graphs}, target\_property\_dim)`。
          - **node_pred** (Tensor) - 节点级预测。形状：:math:`(n_{nodes}, target\_property\_dim)`。
          - **stress_pred** (Tensor) - 应力预测（如果提供stress_head）。形状：:math:`(n_{graphs}, 6)`。

    异常：
        - **ValueError** - 如果既未提供node_head也未提供graph_head。
        - **ValueError** - 如果cutoff_layers超过基础模型中的消息传递步骤数。
        - **ValueError** - 如果graph_head需要时未提供atomic_numbers。