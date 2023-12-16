sponge.colvar.Volume
========================

.. py:class:: sponge.colvar.Volume(name: str = 'volume')

    模拟系统的体积。

    参数:
        - **name** (str) - Colvar的名称。默认值:'volume'。

    .. py:method:: construct(coordinate: Tensor, pbc_box: bool = None)

        得到常量值。

        参数:
            - **coordinate** (Tensor) - 张量的shape为 (B, A, D) 。数据类型为float。
            - **pbc_box** (Tensor) - 张量的shape为 (B, D) 。数据类型为float。默认值：``None``。

        返回：
            体积(Tensor): 张量的shape为 (B, ...) 或 (B, ..., 1) 。数据类型为浮点型。