mindchemistry.e3.nn.FullyConnectedNet
=====================================

.. py:class:: mindchemistry.e3.nn.FullyConnectedNet(h_list, act=None, out_act=False, init_method='normal', dtype=float32)

    带有标量归一化激活的全连接神经网络。

    参数：
        - **h_list** (List[int]) - 用于密集层的输入、内部和输出维度的列表。
        - **act** (Func) - 将自动归一化的激活函数。默认值：``None``。
        - **out_act** (bool) - 是否对输出应用激活函数。默认值：``False``。
        - **init_method** (Union[str, mindspore.common.initializer]) - 初始化参数的方法。默认值：``'normal'`` 。
        - **dtype** (mindspore.dtype) - 输入张量的类型。默认值：``mindspore.float32`` 。

    输入：
        - **input** (Tensor) - 形状为 :math:`(h_list[0])` 的张量。

    输出：
        - **output** (Tensor) - 形状为 :math:`(h_list[-1])` 的张量。

    异常：
        - **TypeError**: 如果 `h_list` 的元素不是 `int`。
