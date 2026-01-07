mindscience.common.to_2tuple
=============================

.. py:function:: mindscience.common.to_2tuple(t)

    将一个整数或整数元组转换为长度为 ``2`` 的元组。

    参数：
        - **t** (Union[int, tuple(int)]) - 网格的高度和宽度。

    返回：
        Tuple(int, int)，与输入相同，或者是一个 `(t, t)` 形式的元组。