sciai.utils.to_tensor
=======================

.. py:function:: sciai.utils.to_tensor(tensors, dtype=ms.float32)

    将数组/张量转换为给定的MindSpore数据类型。

    参数：
        - **tensors** (Union[Tensor, ndarray, Number, np.floating, tuple[Tensor, ndarray]]) - 要转换的若干Tensor。
        - **dtype** (type) - 目标Mindspore Tensor数据类型。 默认值：ms.float32。

    返回：
        Union(Tensor, tuple(Tensor)) - 单个或元组张量。

    异常：
        - **TypeError** 如果输入类型不正确。