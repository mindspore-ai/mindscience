sciai.operators.grad
=======================

.. py:function:: sciai.operators.grad(net, output_index=0, input_index=-1)

    根据给定的输出索引和输入索引，获取指定网络的导函数网络。所有输出索引将用于求导并结果求和，所有输入索引将分别求导。

    参数：
        - **net** (Cell) - 用于自动微分的网络。
        - **output_index** (int) - 输出索引，从0开始计数。默认值：0。
        - **input_index** (Union(int, tuple[int])) - 需要求导的输入索引，从0开始计数，只允许正向索引。若为-1，则所有指定输入将用于分别求导。默认值：-1。

    输入：
        - **\*inputs** (tuple[Tensor]) - 原网络的输入。

    输出：
        Union(Tensor, tuple[Tensor])，一阶导函数网络的输出。

    异常：
        - **TypeError** - 如果 out_index 不是 int。
        - **TypeError** - 如果 input_index 既不是 `int` 也不是 `int` 的元组/列表。
        - **TypeError** - 如果神经网络的输出既不是 `Tensor` ，也不是 `Tensor` 的元组。
        - **TypeError** - 如果 `input_index` 类型既不是 `int` 也不是 `int` 的元组。
        - **IndexError** - 如果 `input_index` 或 `output_index` 超出范围。