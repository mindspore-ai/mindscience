mindscience.common.to_3tuple
=============================

.. py:function:: mindscience.common.to_3tuple(t)

    将一个整数或整数元组转换为长度为 ``3`` 的元组。

    参数：
        - **t** (Union[int, tuple(int)]) - 网格的高度、宽度和深度。

    返回：
        Tuple(int, int, int)，与输入相同，或者是一个 `(t, t, t)` 形式的元组。