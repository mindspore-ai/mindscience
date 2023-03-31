mindsponge.metrics.sidechain
============================

.. py:function:: mindsponge.metrics.sidechain(alt_naming_is_better, rigidgroups_gt_frames, rigidgroups_alt_gt_frames, rigidgroups_gt_exists, renamed_atom14_gt_positions, renamed_atom14_gt_exists, sidechain_atom_clamp_distance, sidechain_length_scale, pred_frames, pred_positions)

    调用 `frame_aligned_point_error` 实现全原子损失函数计算
    `Jumper et al. (2021) Suppl. Alg. 20 "StructureModule" line 17
    <https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-021-03819-2/MediaObjects/41586_2021_3819_MOESM1_ESM.pdf>`_.

    参数：
        - **alt_naming_is_better** (Tensor) - shape为 :math:`(N_{res},)` 的tensor，如果值为1，该残基使用备选的真实值计算损失函数更好。
        - **rigidgroups_gt_frames** (Tensor) - 真实局部坐标系，shape为 :math:`(N_{res}, 8, 12)` ，:math:`N_{res}` 是蛋白质中的残基数目。每个残基具有一个骨架刚体变换组和7个由扭转角定义的刚体变换组（这里包括三个主链扭转角和四个侧链扭转角）；对于最后一维，前九个元素代表局部坐标系相对于全局坐标系的旋转矩阵；后三个代表局部坐标系相对于全局坐标系的平移矩阵。
        - **rigidgroups_alt_gt_frames** (Tensor) - 备选的真实局部坐标系，部分氨基酸存在手性，因此使用手性对称的位置作为备选。 shape与rigidgroups_gt_frames一致。
        - **rigidgroups_gt_exists** (Tensor) - 真实局部坐标系的mask，shape为 :math:`(N_{res}, 8)` 。
        - **renamed_atom14_gt_positions** (Tensor) - 重命名的真实原子坐标（部分氨基酸存在手性对称，需先进行对称变换操作重命名，参见函数 `compute_renamed_ground_truth`），采用14位稠密编码，shape为 :math:`(N_{res}, 14, 3)` 。
        - **renamed_atom14_gt_exists** (Tensor) - 重命名后的原子坐标的mask，shape为 :math:`(N_{res}, 14)` 。
        - **sidechain_atom_clamp_distance** (float) - 距离误差的截断点，超过该距离时梯度不再考虑，常量。
        - **sidechain_length_scale** (float) - 单位距离，用于缩放距离的差，常量。
        - **pred_frames** (Tensor) - 预测的局部坐标系，shape为 :math:`(12, N_{recycle}, N_{res}, 8)` ，其中 :math:`N_{recycle}` 是Structure模块中FoldIteration的循环次数，实际只使用最后一次循环产生的局部坐标系。
        - **pred_positions** (Tensor) - 预测的原子坐标，shape为 :math:`(3, N_{recycle}, N_{res}, 14)` ，实际只使用最后一次循环产生的坐标。

    返回：
        fape(Tensor)。计算所得全原子点位置误差，计算过程中过大的误差会被截断。shape为 :math:`()` 。