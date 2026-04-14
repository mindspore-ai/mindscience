sponge.partition.IndexDistances
================================

.. py:class:: sponge.partition.IndexDistances(use_pbc: bool = None, large_dis: float = 100, keepdims: bool = False)

    计算相邻原子之间的距离。

    参数：
        - **use_pbc** (bool，可选) - 是否使用周期性边界条件。默认值： ``None``。
        - **large_dis** (float，可选) - 一个大值，添加到距离等于零的值以防止它们在 Norm 操作后变成零值，这可能导致自动微分错误。默认值： ``100.0``。
        - **keepdims** (bool，可选) - 如果为 ``True``，则结果中最后一个轴将被保留为大小为一的维度。默认值： ``False``。
