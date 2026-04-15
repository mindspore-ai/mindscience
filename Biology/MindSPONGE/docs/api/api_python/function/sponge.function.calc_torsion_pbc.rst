sponge.function.calc_torsion_pbc
=========================================

.. py:function:: sponge.function.calc_torsion_pbc(position_a: Tensor, position_b: Tensor, position_c: Tensor, position_d: Tensor, pbc_box: Tensor, keepdims: bool = False)

    在有周期性边界条件的情况下计算由四个位置A，B，C，D形成的扭转角。

    参数：
        - **position_a** (Tensor) - 位置a，shape为 :math:`(..., D)` ，D是模拟系统的维度。
        - **position_b** (Tensor) - 位置b，shape为 :math:`(..., D)` 。
        - **position_c** (Tensor) - 位置c，shape为 :math:`(..., D)` 。
        - **position_d** (Tensor) - 位置d，shape为 :math:`(..., D)` 。
        - **pbc_box** (Tensor) - PBC box，shape为 :math:`(D)` 或 :math:`(B, D)` ，B是Batch size。
        - **keepdims** (bool) - 设置为 ``True`` 的话，最后一个维度会保留，默认值 ``False`` 。

    返回：
        Tensor。计算所得扭转角。shape为 :math:`(...)` 或 :math:`(..., 1)` 。
