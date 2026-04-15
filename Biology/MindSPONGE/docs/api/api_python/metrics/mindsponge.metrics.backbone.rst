mindsponge.metrics.backbone
==========================================

.. py:function:: mindsponge.metrics.backbone(traj, backbone_affine_tensor, backbone_affine_mask, fape_clamp_distance, fape_loss_unit_distance, use_clamped_fape)

    调用 `frame_aligned_point_error_map` 实现骨架全原子损失函数计算
    `Jumper et al. (2021) Suppl. Alg. 20 "StructureModule" line 17
    <https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-021-03819-2/MediaObjects/41586_2021_3819_MOESM1_ESM.pdf>`_.

    参数：
        - **traj** (Tensor) - Structure模块预测的一系列骨架局部坐标系（轨迹），shape为 :math:`(N_{recycle}, N_{res}, 7)` ，其中 :math:`N_{recycle}` 是Structure模块中的循环次数， :math:`N_{res}` 是蛋白质中的残基数目，对于最后一维，前四个分量是表征旋转的四元数，代表局部坐标系相对全局坐标系的旋转，后三个分量是三维空间的平移。
        - **backbone_affine_tensor** (Tensor) - 真实的的骨架局部坐标系，shape为 :math:`(N_{res}, 7)` 。
        - **backbone_affine_mask** (Tensor) - 骨架局部坐标系的mask，shape为 :math:`(N_{res},)` 。
        - **fape_loss_unit_distance** (float) - 单位距离，用于缩放距离的差，常量。
        - **fape_clamp_distance** (float) - 距离误差的截断点，超过该距离时梯度不再考虑，常量。
        - **use_clamped_fape** (float) - 是否截断截断距离误差， ``0`` 或者 ``1``， ``0`` 代表不截断。

    返回：
        - **fape** (list) - Tensor。计算所得Structure模块最后一次迭代输出的结构的全原子点位置误差，如果use_clamped_fape为1，则计算过程中过大的误差会被截断。shape为 :math:`()` 。
        - **loss** (list) - Tensor。计算所得Structure模块所有迭代输出的结构的全原子点位置误差的均值，如果use_clamped_fape为1，则计算过程中过大的误差会被截断。shape为 :math:`()` 。
        - **no_clamp** (list) - Tensor。计算所得Structure模块最后一次迭代输出的结构的全原子点位置误差，没有截断。shape为 :math:`()` 。