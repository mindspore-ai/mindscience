mindchemistry.e3.o3.wigner_3j
==============================

.. py:function:: mindchemistry.e3.o3.wigner_3j(l1, l2, l3, dtype)

    Wigner 3j符号 :math:`C_{lmn}`。
    它满足以下两个性质:

    .. math::
        C_{lmn} = C_{ijk} D_{il}(g) D_{jm}(g) D_{kn}(g) \qquad \forall g \in SO(3)

    其中 :math:`D` are given by `wigner_D`.

    .. math::
        C_{ijk} C_{ijk} = 1

    参数:
        - **l1** (int) - :math:`l_1`。
        - **l2** (int) - :math:`l_2`。
        - **l3** (int) - :math:`l_3`。
        - **dtype** (mindspore.dtype) - 输入张量的类型。默认值：``mindspore.float32`` 。

    返回:
        - **output** (Tensor) - 张量，Wigner 3j符号 :math:`C_{lmn}`。形状为 :math:`(2l_1+1, 2l_2+1, 2l_3+1)` 的张量。
