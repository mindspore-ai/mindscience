mindscience.distributed.modules.RowParallelLinear
=================================================

.. py:class:: RowParallelLinear(in_features, out_features, bias=True, input_is_parallel=False, use_sequence_parallel=False, weight_init=None, bias_init=None, param_init_dtype=ms.float32, compute_dtype=ms.bfloat16)

    行并行线性层，将输入特征维度在TP通信组中进行分片。

    参数：
        - **in_features** (int) - TP通信组内所有卡的总输入特征数量。
        - **out_features** (int) - 输出特征数量。
        - **bias** (bool, 可选) - 是否创建偏置参数（不沿输入维度分片）。默认值：``True``。
        - **input_is_parallel** (bool, 可选) - 是否期望输入已在TP通信组中切分。默认值：``False``。
        - **use_sequence_parallel** (bool, 可选) - 是否对输出使用reduce scatter而不是all reduce。默认值：``False``。
        - **weight_init** (Union[Initializer, str], 可选) - 权重初始化方法。默认值：``None``。
        - **bias_init** (Union[Initializer, str], 可选) - 偏置初始化方法。默认值：``None``。
        - **param_init_dtype** (mstype.dtype, 可选) - 参数初始化数据类型。默认值：``ms.float32``。
        - **compute_dtype** (mstype.dtype, 可选) - 计算数据类型。默认值：``ms.bfloat16``。

    输入：
        - **x** (Tensor) - 形状为 (seq_len, in_features // TP) 或 (seq_len, in_features) 的输入张量，具体取决于 `input_is_parallel` 是否为 ``True``。

    输出：
        - **output** (Tensor) - 形状为 (seq_len // TP, out_features) 或 (seq_len, out_features) 的输出张量，具体取决于 `use_sequence_parallel` 是否为 ``True``。
