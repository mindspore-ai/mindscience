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

    .. py:method:: construct(initial: Tensor, terminal: Tensor, pbc_box: Tensor = None)

        计算从初始点到终点的向量。

        参数：
            - **initial** (Tensor) - 初始点的坐标。张量的shape为 :math:`(B, ..., D)` 。数据类型为float。
              其中，B表示batchsize，例如，模拟中的步行者数量。D表示仿真系统的空间维度。通常为3。
            - **terminal** (Tensor) - 终止位置的坐标。张量的shape为 :math:`(B, ..., D)` 。数据类型为float。
            - **pbc_box** (Tensor) - 张量的shape为 :math:`(B, D)` 。数据类型为float。默认值： ``None``。

        返回：
            Tensor。距离。张量的shape为 :math:`(B, ...)` 。数据类型为float。
