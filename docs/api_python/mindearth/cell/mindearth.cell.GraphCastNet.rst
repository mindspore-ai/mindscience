mindearth.cell.GraphCastNet
============================

.. py:class:: mindearth.cell.GraphCastNet(vg_in_channels, vg_out_channels, vm_in_channels, em_in_channels, eg2m_in_channels, em2g_in_channels, latent_dims, processing_steps, g2m_src_idx, g2m_dst_idx, m2m_src_idx, m2m_dst_idx, m2g_src_idx, m2g_dst_idx, mesh_node_feats, mesh_edge_feats, g2m_edge_feats, m2g_edge_feats, per_variable_level_mean, per_variable_level_std, recompute=False)

    GraphCast 基于一种新颖的基于图神经网络的高分辨率多尺度网格表示自回归模型。
    有关更多详细信息，请参考论文 `GraphCast: Learning skillful medium-range global weather forecasting <https://arxiv.org/pdf/2212.12794.pdf>`_ 。

    参数：
        - **vg_in_channels** (int) - grid网格节点输入尺寸。
        - **vg_out_channels** (int) - 网格节点输出尺寸。
        - **vm_in_channels** (int) - mesh网格节点输入尺寸。
        - **em_in_channels** (int) - 网格边缘尺寸。
        - **eg2m_in_channels** (int) - grid网格到mesh网格边缘尺寸。
        - **em2g_in_channels** (int) - mesh网格到grid网格边缘尺寸。
        - **latent_dims** (int) - 隐藏层的dim数量。
        - **processing_steps** (int) - 处理的步骤数。
        - **g2m_src_idx** (Tensor) - grid网格到mesh网格边的源节点索引。
        - **g2m_dst_idx** (Tensor) - grid网格到mesh网格边的目标节点索引。
        - **m2m_src_idx** (Tensor) - mesh网格源节点到mesh网格边的索引。
        - **m2m_dst_idx** (Tensor) - mesh网格到mesh网格边的目标节点索引。
        - **m2g_src_idx** (Tensor) - mesh网格到grid网格边的源节点索引。
        - **m2g_dst_idx** (Tensor) - mesh网格到grid网格边的目标节点索引。
        - **mesh_node_feats** (Tensor) - 网格节点的特征。
        - **mesh_edge_feats** (Tensor) - 网格边缘的特征。
        - **g2m_edge_feats** (Tensor) - grid网格到mesh网格边的特征。
        - **m2g_edge_feats** (Tensor) - mesh网格到grid网格边的特征。
        - **per_variable_level_mean** (Tensor) - 每个变量特定尺度的平均值。
        - **per_variable_level_std** (Tensor) - 每个变量特定尺度的方差。
        - **recompute** (bool, optional) - 设置是否重计算。 默认值： ``False`` 。

    输入：
        - **input** (Tensor) - shape为 :math:`(batch\_size, height\_size * width\_size, feature\_size)` 的Tensor。

    输出：
        Tensor，Graphcast网络的输出。

        - **output** (Tensor) - shape为 :math:`(height\_size * width\_size, feature\_size)` 的Tensor。