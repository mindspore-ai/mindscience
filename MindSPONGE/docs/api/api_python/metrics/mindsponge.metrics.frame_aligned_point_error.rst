mindsponge.metrics.frame_aligned_point_error
===============================================

.. py:function:: mindsponge.metrics.frame_aligned_point_error(pred_frames, target_frames, frames_mask, pred_positions, target_positions, positions_mask, length_scale, l1_clamp_distance)

    在不同的局部坐标系下计算两个结构的原子位置误差。
    将预测的原子坐标映射到不同的预测局部坐标系下：
    :math:`\vec{x_{j\_pred}^{i}} = \mathcal{T}_{i\_{pred}} \circ \vec{x_{j\_pred}}`
    同时将真实的原子坐标映射到对应的真实局部坐标系下：
    :math:`\vec{x_{j\_gt}^{i}} = \mathcal{T}_{i\_{gt}} \circ \vec{x_{j\_gt}}`
    然后两两计算结构误差取所有局部坐标系中所有原子坐标L2误差的均值：
    :math:`\sum_{i }^{N_{frames}}\sum_{j}^{N_{atoms}}(\parallel \vec{x_{j\_pred}^{i}} - \vec{x_{j\_gt}^{i}} \parallel )` 

    参数：
        - **pred_frames** (Tensor) - 预测的蛋白质刚体变换组对应局部坐标系，shape为 :math:`(12, N_{frames})` ，其中 :math:`N_{frames}` 是局部坐标系的数量。其中第一维上前九个元素代表局部坐标系相对于全局坐标系的旋转矩阵；后三个代表局部坐标系相对于全局坐标系的平移矩阵。
        - **target_frames** (Tensor) - 预测的蛋白质刚体变换组对应局部坐标系，shape与pred_frames一致。
        - **frames_mask** (Tensor) - 局部坐标系的mask，shape为 :math:`(N_{frames},)` 。
        - **pred_positions** (Tensor) - 预测的原子坐标，shape为 :math:`(3, N_{atoms})` 。
        - **target_positions** (Tensor) - 真实的原子坐标，shape与pred_positions一致。
        - **positions_mask** (Tensor) - 预测的原子坐标的mask，shape为 :math:`(N_{atoms},)`  。
        - **length_scale** (float) - 单位距离，用于缩放距离的差，常量。
        - **l1_clamp_distance** (float) - 距离误差的截断点，超过该距离时梯度不再考虑，常量。

    返回：
        - **error_clamp** (Tensor) - Tensor。计算所得全原子点位置误差，计算过程中过大的误差会被截断。shape为 :math:`(N_{recycle},)` 。