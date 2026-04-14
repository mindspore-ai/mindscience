sponge.function.coulomb_interaction
=======================================

.. py:function:: sponge.function.coulomb_interaction(q_i: Tensor, q_j: Tensor, r_ij: Tensor, mask: Tensor = None, coulomb_const: float = 1)

    计算库仑相互作用。

    .. math::

        E_{coulomb}(r_{ij}) = k \frac{q_i q_j}{r_{ij}}

    参数：
        - **q_i** (Tensor) - 张量的shape为 :math:`(...)`。数据类型为float。原子 :math:`q_i`的电荷 :math:`i`-th 。
        - **q_j** (Tensor) - 张量的shape为 :math:`(...)`。数据类型为float。原子 :math:`q_j`的电荷 :math:`j`-th 。
        - **r_ij** (Tensor) - 张量的shape为 :math:`(...)`。数据类型为float。原子 :math:`i` 和 :math:`i` 之间的距离 :math:`r_{ij}`。
        - **mask** (Tensor) - 张量的shape为 :math:`(...)`。数据类型为bool。距离 :math:`r_{ij}` 的掩码。默认值： ``None``。数据类型为bool。距离
        - **coulomb_const** (float) - 库仑常量 :math:`k` 。默认值：1
    
    返回：
        Tensor。E_coulomb。张量的shape为 :math:`(...)` 。数据类型为float。
    
