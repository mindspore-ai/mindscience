sponge.function.periodic_difference
=======================================

.. py:function:: sponge.function.periodic_difference(difference: Tensor, period: Tensor, mask: Tensor = None, offset: float = -0.5)

    获取周期变量之间的差值。

    参数：
        - **variable** (Tensor) - 张量的shape为 (...) 。数据类型为float。周期变量
        - **period** (Tensor) - 张量的shape为 (...) 。数据类型为float。周期的上边界。
        - **mask** (Tensor) - 张量的shape为 (...) 。数据类型为bool。周期变量的掩码。
        - **offset** (float) - 偏移比 :math:`c` 与周期 :math:`theta` 的相对距离。默认值：-0.5
    返回：
        Tensor。period_diff。张量的shape为 (...) 。数据类型为float。值在周期范围内的变量。

