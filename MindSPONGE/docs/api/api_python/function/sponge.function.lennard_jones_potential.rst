sponge.function.lennard_jones_potential
===========================================

.. py:function:: sponge.function.lennard_jones_potential(epsilon: Tensor, sigma: Tensor, r_ij: Tensor, mask: Tensor = None)

    使用 :math:`\epsilon` 和 :math:`\sigma` 计算Lennard-Jones (LJ) 势。

    .. math::

        E_{lj}(r_{ij}) = 4 \epsilon \left [\left ( \frac{\sigma}{r_{ij}} \right ) ^{12} -
                                           \left ( \frac{\sigma}{r_{ij}} \right ) ^{6} \right]

    参数：
        - **epsilon** (Tensor) - 张量的shape为 :math:`(...)` 。数据类型为float。井深 :math:`\epsilon`。
        - **sigma** (Tensor) - 张量的shape为 :math:`(...)` 。数据类型为float。特征距离 :math:`\sigma`。
        - **r_ij** (Tensor) - 张量的shape为 :math:`(...)` 。数据类型为float。原子 :math:`i` 和 :math:`i` 之间的距离 :math:`r_{ij}`。
        - **mask** (Tensor) - 张量的shape为 :math:`(...)` 。数据类型为bool。距离的掩码 :math:`r_{ij}`。默认值： ``None``。

    返回：
        Tensor。E_coulomb。张量的shape为 :math:`(...)` 。数据类型为float。

