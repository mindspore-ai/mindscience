sponge.function.GetDistance
===============================

.. py:class:: sponge.function.GetDistance(use_pbc: bool = None, keepdims: bool = False, axis: int = -1)

    获取有或者没有PBC box的距离。

    参数：
        - **use_pbc** (bool) - 计算距离时是否使用周期性边界条件。
          如果给出 "None" ，它将会根据是否给出了pbc_box决定计算距离时是否使用周期性边界条件。默认值： ``None`` 。
        - **keepdims** (bool) - 是否将输出张量的最后一个维度保持在范数之后的距离。
          如果给出 "True" ，输出张量的最后一个维度将为1。默认值： ``False`` 。
        - **axis** (int) - 计算距离时是否使用周期性边界条件。默认值： ``-1`` 。
