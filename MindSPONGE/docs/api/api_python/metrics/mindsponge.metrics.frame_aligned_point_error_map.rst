mindsponge.metrics.frame_aligned_point_error_map
===========================================================

.. py:function:: mindsponge.metrics.frame_aligned_point_error_map(pred_frames, target_frames, frames_mask, pred_positions, target_positions, positions_mask, length_scale, l1_clamp_distance)

    在不同的局部坐标系下计算两个结构的原子位置误差，与 `frame_aligned_point_error` 函数相似，区别在于带批处理逻辑，同时计算多组局部坐标系与真实结构局部坐标系之间的误差，针对每组局部坐标系分别返回一个损失函数值，且只考虑 :math:`C\alpha` 原子，计算逻辑参考 `frame_aligned_point_error`。

    参数：
        - **pred_frames** (list) - 预测的蛋白质刚体变换组对应局部坐标系，二维数组，数组的第一个元素是长度为9的tensor的list，代表局部坐标系相对于全局坐标系的旋转矩阵；第二个元素是长度为3的tensor的list，代表局部坐标系相对于全局坐标系的平移矩阵，所有tensor的shape均为 :math:`(N_{recycle}, N_{res})` ，其中 :math:`N_{recycle}` 是Structure模块中FoldIteration的循环次数。 :math:`N_{res}` 是蛋白质中的残基数目。
        - **target_frames** (list) - 预测的蛋白质刚体变换组对应局部坐标系，也是二维list，shape与 `pred_frames` 一致，为 :math:`(N_{res},)`。
        - **frames_mask** (Tensor) - 局部坐标系的mask，shape为 :math:`(N_{res},)` 。
        - **pred_positions** (list) - 预测的 :math:`C\alpha` 原子的坐标，长度为3的tensor的一维数组，tensor的shape为 :math:`(N_{recycle}, N_{res},)` 。
        - **target_positions** (list) - 真实的 :math:`C\alpha` 原子的坐标，shape为 :math:`(N_{res},)` 的 3 个Tensor的list。
        - **positions_mask** (Tensor) - 预测的原子坐标的mask，shape为 :math:`(N_{res},)` 。
        - **length_scale** (float) - 单位距离，用于缩放距离的差，常量。
        - **l1_clamp_distance** (float) - 距离误差的截断点，超过该距离时梯度不再考虑，常量。

    返回：
        - **error_clamp** (Tensor) - Tensor。计算所得骨架全原子点位置误差，计算过程中过大的误差会被截断。shape为 :math:`(N_{recycle},)` 。
        - **error_no_clamp** (Tensor) - Tensor。计算所得骨架原子点位置误差（没有截断）。shape为 :math:`(N_{recycle},)` 。