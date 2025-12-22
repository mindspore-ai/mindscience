mindscience.distributed.mappings.gather_from_sequence
======================================================

.. py:function:: gather_from_sequence(x, group, tensor_parallel_output_grad=True)

    沿第一个维度收集切分的张量。

    参数：
        - **x** (Tensor) - 沿第一个维度具有序列分区的输入张量。
        - **group** (Union[CommGroup, CommGroupBase]) - 操作的通信组。
        - **tensor_parallel_output_grad** (bool, 可选) - 确定在反向传播中是使用
          reduce-scatter（True）还是 scatter（False）的标志。默认值：``True``。

    返回：
        沿第一个维度收集所有切分的张量。
