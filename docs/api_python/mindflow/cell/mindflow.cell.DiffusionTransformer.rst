mindflow.cell.DiffusionTransformer
==================================================

.. py:class:: mindflow.cell.DiffusionTransformer(in_channels, out_channels, hidden_channels, layers, heads, time_token_cond=True, compute_dtype=mstype.float32)

    以Transformer作为骨干网络的扩散模型。

    参数：
        - **in_channels** (int) - 输入特征维度。
        - **out_channels** (int) - 输出特征维度。
        - **hidden_channels** (int) - 隐藏层特征维度。
        - **layers** (int) - `Transformer` 层数。
        - **heads** (int) - 注意力头数。
        - **time_token_cond** (bool) - 是否将时间作为作为条件token。 Default: ``True`` 。
        - **compute_dtype** (mindspore.dtype) 计算数据类型。支持 ``mstype.float32`` or ``mstype.float16`` 。 默认值: ``mstype.float32`` ，表示 ``mindspore.float32`` 。

    输入：
        - **x** (Tensor) - 网络输入。shape为 :math:`(batch\_size, sequence\_len, in\_channels)` 的Tensor。
        - **timestep** (Tensor) - 时间步。shape为 :math:`(batch\_size,)` 的Tensor。

    输出：
        - **output** (Tensor) - shape为 :math:`(batch\_size, sequence\_len, out\_channels)` 的Tensor。
