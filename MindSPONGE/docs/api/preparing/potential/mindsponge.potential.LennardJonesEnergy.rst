mindsponge.potential.LennardJonesEnergy
=======================================

.. py:class:: mindsponge.potential.LennardJonesEnergy(epsilon=None, sigma=None, mean_c6=0, parameters=None, cutoff=None, use_pbc=None, length_unit="nm", energy_unit="kj/mol", units=None)

    Lennard-Jones势。

    .. math::

        E_{lj}(r_{ij}) = 4 * \epsilon_{ij} * [(\sigma_{ij} / r_{ij}) ^ {12} - (\sigma_{ij} / r_{ij}) ^ 6]

        \epsilon_{ij} = \sqrt(\epsilon_i * \epsilon_j)

        \sigma_{ij} = 1 / 2 * (\sigma_i + \sigma_j)

    参数：
        - **epsilon** (Tensor) - LJ势的 :math:`\epsilon` 参数，shape为(B, A)。默认值："None"。
        - **sigma** (Tensor) - LJ势的:math:`\sigma` 参数，shape为(B, A)。默认值："None"。
        - **mean_c6** (Tensor) - 用于长距离校正色散相互作用的系统的平均色散(<C6>)，shape为(B, A)。默认值：0。
        - **parameters** (dict) - 力场参数。默认值："None"。
        - **cutoff** (float) - 中断距离。默认值："None"。
        - **use_pbc** (bool) - 是否使用PBC。默认值："None"。
        - **length_unit** (str) - 位置坐标的长度单位。默认值："nm"。
        - **energy_unit** (str) - 能量单位。默认值："kj/mol"。
        - **units** (Units) - 长度和能量单位。默认值："None"。

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