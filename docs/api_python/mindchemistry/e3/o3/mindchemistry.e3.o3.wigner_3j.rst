mindchemistry.e3.o3.wigner_3j
==============================

.. py:function:: mindchemistry.e3.o3.wigner_3j(l1, l2, l3, dtype=float32)

    Wigner 3j符号 :math:`C_{lmn}`。
    它满足以下两个性质:

    .. math::
        C_{lmn} = C_{ijk} D_{il}(g) D_{jm}(g) D_{kn}(g) \qquad \forall g \in SO(3)

    其中 :math:`D` 由 `wigner_D` 给出。

    .. math::
        C_{ijk} C_{ijk} = 1

    参数：
        - **l1** (int) - ``wigner_3j`` 的 :math:`l_1` 参数。
        - **l2** (int) - ``wigner_3j`` 的 :math:`l_2` 参数。
        - **l3** (int) - ``wigner_3j`` 的 :math:`l_3` 参数。
        - **dtype** (mindspore.dtype) - 输入张量的类型。默认值：``mindspore.float32`` 。

    返回：
        - **output** (Tensor) - 张量，Wigner 3j符号 :math:`C_{lmn}`。形状为 :math:`(2l_1+1, 2l_2+1, 2l_3+1)` 的张量。

    异常：
        - **TypeError** - 如果 `l1`、 `l2` 或 `l3` 不是整型。
        - **ValueError** - 如果 `l1`、 `l2` 和 `l3` 不满足 `abs(l2 - l3) <= l1 <= l2 + l3` 。
