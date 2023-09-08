mindearth.cell.GraphCastNet
============================

.. py:class:: mindearth.cell.GraphCastNet(vg_in_channels, vg_out_channels, vm_in_channels, em_in_channels, eg2m_in_channels, em2g_in_channels, latent_dims, processing_steps, g2m_src_idx, g2m_dst_idx, m2m_src_idx, m2m_dst_idx, m2g_src_idx, m2g_dst_idx, mesh_node_feats, mesh_edge_feats, g2m_edge_feats, m2g_edge_feats, per_variable_level_mean, per_variable_level_std, recompute=False)

    GraphCast是基于图神经网络和一种新的高分辨率多尺度网格表示的自回归模型。详细信息可在 `GraphCast：全球中期天气预报中找到 <https://arxiv.org/pdf/2212.12794.pdf>`_ 。

    参数：
        - **vg_in_channels** (int) - grid网格节点的维度。
        - **vg_out_channels** (int) - grid网格节点的最终维度。
        - **vm_in_channels** (int) - mesh网格节点的维度。
        - **em_in_channels** (int) - mesh网格边缘的维度。
        - **eg2m_in_channels** (int) - grid网格到mesh网格边缘的维度。
        - **em2g_in_channels** (int) - mesh网格到grid网格边缘的维度。
        - **latent_dims** (int) - 隐藏层数。
        - **processing_steps** (int) - 处理步骤数。
        - **g2m_src_idx** (Tensor) - grid网格到mesh网格边的源节点索引。
        - **g2m_dst_idx** (Tensor) - grid网格到mesh网格边的目标节点索引。
        - **m2m_src_idx** (Tensor) - mesh网格到mesh网格边的源节点索引。
        - **m2m_dst_idx** (Tensor) - mesh网格到mesh网格边的目标节点索引。
        - **m2g_src_idx** (Tensor) - mesh网格到grid网格边的源节点索引。
        - **m2g_dst_idx** (Tensor) - mesh网格到grid网格边的目标节点索引。
        - **mesh_node_feats** (Tensor) - mesh网格节点的特征。
        - **mesh_edge_feats** (Tensor) - mesh网格边的特征。
        - **g2m_edge_feats** (Tensor) - grid网格到mesh网格边的特征。
        - **m2g_edge_feats** (Tensor) - mesh网格到grid网格边的特征。
        - **per_variable_level_mean** (Tensor) - 随时间变化的每个水平上变量的方差的平均值。
        - **per_variable_level_std** (Tensor) - 随时间变化的每个水平上变量的标准差。
        - **recompute** (bool, 可选) - 确定是否重计算。默认值：False。

    输入：
        - **grid_node_feats** (Tensor) - shape为 :math:`(batch\_size, height\_size * width\_size, feature\_size)` 的Tensor。

    输出：
        - **output** (Tensor) - shape为 :math:`(height\_size * width\_size, feature\_size)` 的Tensor。