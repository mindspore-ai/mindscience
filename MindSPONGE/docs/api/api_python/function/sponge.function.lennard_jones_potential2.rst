sponge.function.lennard_jones_potential2
============================================

.. py:function:: sponge.function.lennard_jones_potential2(epsilon: Tensor, r_0: Tensor, r_ij: Tensor, mask: Tensor = None)

    使用 :math:`\epsilon` 和 :math:`r_0` 计算Lennard-Jones (LJ) 势。

    .. math::

        E_{lj}(r_{ij}) = 4 \epsilon \left [\frac{1}{4} \left ( \frac{r_0}{r_{ij}} \right ) ^{12} -
                                           \frac{1}{2} \left ( \frac{r_0}{r_{ij}} \right ) ^{6} \right]
    
    参数：
        - **epsilon** (Tensor) - 张量的shape为 :math:`(...)` 。数据类型为float。井深 :math:`\epsilon`。
        - **r_0** (Tensor) - 张量的shape为 :math:`(...)` 。数据类型为float。原子半径 :math:`r_0`。
        - **r_ij** (Tensor) - 张量的shape为 :math:`(...)` 。数据类型为float。原子 :math:`i` 和 :math:`i` 之间的距离 :math:`r_{ij}`。
        - **mask** (Tensor) - 张量的shape为 :math:`(...)` 。数据类型为bool。距离的掩码 :math:`r_{ij}`。默认值： ``None``。

    返回：
        Tensor。E_coulomb。张量的shape为 :math:`(...)` 。数据类型为float。

