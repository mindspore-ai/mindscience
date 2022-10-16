mindsponge.potential.LennardJonesEnergy
=======================================

.. py:class:: mindsponge.potential.LennardJonesEnergy(epsilon, sigma, mean_c6=0, parameters, cutoff, use_pbc, length_unit="nm", energy_unit="kj/mol", units)

    Lennard-Jones势。

    .. math::

        E_lj(r_ij) = 4 * \epsilon_ij * [(\sigma_ij / r_ij) ^ 12 - (\sigma_ij / r_ij) ^ 6]

        \epsilon_ij = /sqrt(\epsilon_i * \epsilon_j)

        \sigma_ij = 1 / 2 * (\sigma_i + \sigma_j)

    参数：
        - **epsilon** (Tensor) - LJ势的epsilon参数。
        - **sigma** (Tensor) - LJ势的sigma参数。
        - **mean_c6** (Tensor) - 用于长距离校正色散相互作用的系统的平均色散(<C6>)。
        - **parameters** (dict) - 力场参数。
        - **cutoff** (float) - 中断距离。
        - **use_pbc** (bool) - 是否使用PBC。
        - **length_unit** (str) - 位置坐标的长度单位。
        - **energy_unit** (str) - 能量单位。
        - **units** (Units) - 长度和能量单位。

    输出：
        Tensor。能量，shape为(B, 1)。

    符号：
        - **B** - Batch size。
        - **A** - 原子数量。
        - **N** - 邻居原子的最大数量。
        - **D** - 模拟系统的维度。

    .. py:method:: set_cutoff(cutoff)

        设置中断距离。

        参数：
            - **cutoff** (Tensor) - 中断距离。