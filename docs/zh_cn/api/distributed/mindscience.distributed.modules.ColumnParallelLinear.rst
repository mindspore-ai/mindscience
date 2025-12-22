mindscience.distributed.modules.ColumnParallelLinear
====================================================

.. py:class:: ColumnParallelLinear(in_features, out_features, bias=True, gather_output=True, use_sequence_parallel=False, weight_init=None, bias_init=None, param_init_dtype=ms.float32, compute_dtype=ms.bfloat16)

    列并行线性层，将输出特征维度在TP通信组中进行分片。

    参数：
        - **in_features** (int) - 输入特征数量。
        - **out_features** (int) - TP通信组内所有卡的总输出特征数量。
        - **bias** (bool, 可选) - 是否创建并分割与输出分片一致的偏置参数。默认值：``True``。
        - **gather_output** (bool, 可选) - 是否将TP本地输出收集到完整输出中。默认值：``True``。
        - **use_sequence_parallel** (bool, 可选) - 是否对输入使用all gather而不是广播。默认值：``False``。
        - **weight_init** (Union[Initializer, str], 可选) - 权重初始化方法。默认值：``None``。
        - **bias_init** (Union[Initializer, str], 可选) - 偏置初始化方法。默认值：``None``。
        - **param_init_dtype** (mstype.dtype, 可选) - 参数初始化数据类型。默认值：``ms.float32``。
        - **compute_dtype** (mstype.dtype, 可选) - 计算数据类型。默认值：``ms.bfloat16``。

    输入：
        - **x** (Tensor) - 形状为 (seq_len // TP, in_features) 或 (seq_len, in_features) 的输入张量，具体取决于 `use_sequence_parallel` 是否为 ``True``。

    输出：
        - **output** (Tensor) - 形状为 (seq_len, out_features) 或 (seq_len, out_features // TP) 的输出张量，具体取决于 `gather_output` 是否为 ``True``。
