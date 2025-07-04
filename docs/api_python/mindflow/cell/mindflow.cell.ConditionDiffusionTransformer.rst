mindflow.cell.ConditionDiffusionTransformer
==================================================

.. py:class:: mindflow.cell.ConditionDiffusionTransformer(in_channels, out_channels, cond_channels, hidden_channels, layers, heads, time_token_cond=True, cond_as_token=True, compute_dtype=mstype.float32)

    以Transformer作为骨干网络的条件控制扩散模型。

    参数：
        - **in_channels** (int) - 输入特征维度。
        - **out_channels** (int) - 输出特征维度。
        - **hidden_channels** (int) - 隐藏层特征维度。
        - **cond_channels** (int) - 条件特征维度。
        - **layers** (int) - `Transformer` 层数。
        - **heads** (int) - 注意力头数。
        - **time_token_cond** (bool) - 是否将时间作为条件token。Default: ``True`` 。
        - **cond_as_token** (bool) - 是否将条件作为token。Default: ``True`` 。
        - **compute_dtype** (mindspore.dtype) - 计算数据类型。支持 ``mstype.float32`` or ``mstype.float16`` 。默认值: ``mstype.float32`` ，表示 ``mindspore.float32`` 。

    输入：
        - **x** (Tensor) - 网络输入。shape为 :math:`(batch\_size, sequence\_len, in\_channels)` 的Tensor。
        - **timestep** (Tensor) - 时间步。shape为 :math:`(batch\_size,)` 的Tensor。
        - **condition** (Tensor) - 控制条件。shape为 :math:`(batch\_size, cond\_channels)` 的Tensor。

    输出：
        - **output** (Tensor) - shape为 :math:`(batch\_size, sequence\_len, out\_channels)` 的Tensor。
