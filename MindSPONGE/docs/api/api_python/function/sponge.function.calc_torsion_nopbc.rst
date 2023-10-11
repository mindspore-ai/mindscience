sponge.function.calc_torsion_nopbc
============================================

.. py:function:: sponge.function.calc_torsion_nopbc(position_a, position_b, position_c, position_d, keep_dims: bool = False)

    在没有周期性边界条件的情况下计算由四个位置A，B，C，D形成的扭转角。

    参数：
        - **position_a** (Tensor) - 位置a，shape为 :math:`(..., D)` ，D是模拟系统的维度。
        - **position_b** (Tensor) - 位置b，shape为 :math:`(..., D)` 。
        - **position_c** (Tensor) - 位置c，shape为 :math:`(..., D)` 。
        - **position_d** (Tensor) - 位置d，shape为 :math:`(..., D)` 。
        - **keepdims** (bool) - 设置为 ``True`` 的话，最后一个维度会保留，默认值 ``False`` 。

    返回：
        Tensor。计算所得扭转角。shape为 :math:`(...)` 或 :math:`(..., 1)` 。
