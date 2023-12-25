sponge.function.periodic_variable
=====================================

.. py:function:: sponge.function.periodic_variable(variable: Tensor, upper: Tensor, lower: Tensor = 0, mask: Tensor = None)

    获取周期范围内的值。

    参数：
        - **variable** (Tensor) - 张量的shape为 (...) 。数据类型为float。周期变量
        - **upper** (Tensor) - 张量的shape为 (...) 。数据类型为float。周期的上边界。
        - **lower** (Tensor) - 张量的shape为 (...) 。数据类型为float。周期的下边界。默认值：0
        - **mask** (Tensor) - 张量的shape为 (...) 。数据类型为bool。周期变量的掩码。
    
    返回：
        Tensor。period_value。张量的shape为 (...) 。数据类型为float。值在周期范围内的变量。
    
