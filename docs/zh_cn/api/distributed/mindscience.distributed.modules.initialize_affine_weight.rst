mindscience.distributed.modules.initialize_affine_weight
========================================================

.. py:function:: mindscience.distributed.modules.initialize_affine_weight(init_shape, tp_world_size, partition_dim, init_method="XavierUniform", init_dtype=ms.float32)

    为并行处理初始化并（可选）分割权重张量。

    参数：
        - **init_shape** (Tuple[int]) - 要初始化的权重张量的形状。
        - **tp_world_size** (int) - 张量并行维度大小。
        - **partition_dim** (int) - 沿哪个维度分割权重张量。
        - **init_method** (Union[Initializer, str], 可选) - 使用的初始化方法。默认值：``"XavierUniform"``。
        - **init_dtype** (mstype.dtype, 可选) - 初始化权重的数据类型。默认值：``ms.float32``。

    返回：
        Parameter。当 `tp_world_size` 为 ``1`` 时返回完整参数，否则返回每张卡对应的参数切片。
