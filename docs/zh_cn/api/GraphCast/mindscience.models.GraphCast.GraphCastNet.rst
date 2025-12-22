mindscience.models.GraphCast.GraphCastNet
=======================================================

.. py:class:: mindscience.models.GraphCast.GraphCastNet(vg_in_channels, vg_out_channels, vm_in_channels, em_in_channels, eg2m_in_channels, em2g_in_channels, latent_dims, processing_steps, g2m_src_idx, g2m_dst_idx, m2m_src_idx, m2m_dst_idx, m2g_src_idx, m2g_dst_idx, mesh_node_feats, mesh_edge_feats, g2m_edge_feats, m2g_edge_feats, per_variable_level_mean, per_variable_level_std, recompute=False)

    GraphCast 基于图神经网络和新颖的高分辨率多尺度网格表示的自回归模型。
    详情请参阅 `GraphCast: Learning skillful medium-range
    global weather forecasting <https://arxiv.org/pdf/2212.12794.pdf>`_。

    参数：
        - **vg_in_channels** (int) - grid节点维度。
        - **vg_out_channels** (int) - grid节点最终维度。
        - **vm_in_channels** (int) - mesh节点维度。
        - **em_in_channels** (int) - mesh边维度。
        - **eg2m_in_channels** (int) - grid到mesh边维度。
        - **em2g_in_channels** (int) - mesh到grid边维度。
        - **latent_dims** (int) - 隐藏层的维度数。
        - **processing_steps** (int) - 处理步骤数。
        - **g2m_src_idx** (Tensor) - grid到mesh边的源节点索引。
        - **g2m_dst_idx** (Tensor) - grid到mesh边的目标节点索引。
        - **m2m_src_idx** (Tensor) - mesh到mesh边的源节点索引。
        - **m2m_dst_idx** (Tensor) - mesh到mesh边的目标节点索引。
        - **m2g_src_idx** (Tensor) - mesh到grid边的源节点索引。
        - **m2g_dst_idx** (Tensor) - mesh到grid边的目标节点索引。
        - **mesh_node_feats** (Tensor) - mesh节点特征。
        - **mesh_edge_feats** (Tensor) - mesh边特征。
        - **g2m_edge_feats** (Tensor) - grid到mesh边特征。
        - **m2g_edge_feats** (Tensor) - mesh到grid边特征。
        - **per_variable_level_mean** (Tensor) - 时间差分的每个变量级别的反方差均值。
        - **per_variable_level_std** (Tensor) - 时间差分的每个变量级别的反方差标准差。
        - **recompute** (bool, 可选) - 确定是否重新计算。默认值： ``False`` 。

    输入：
        - **input** (Tensor) - 形状为 :math:`(batch\_size, height\_size * width\_size, feature\_size)` 的张量。

    输出：
        - **output** (Tensor) - 形状为 :math:`(height\_size * width\_size, feature\_size)` 的张量。
