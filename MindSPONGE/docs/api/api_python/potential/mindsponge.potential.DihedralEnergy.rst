mindsponge.potential.DihedralEnergy
===================================

.. py:class:: mindsponge.potential.DihedralEnergy(index, force_constant, periodicity, phase, parameters, use_pbc, energy_unit="kj/mol", units)

    二面角(扭转)的能量项。

    .. math::

        E_dihedral(\omega) = \sum_n 1 / 2 * V_n * [1 - cos(n * \omega - {\gamma}_n)]

    参数：
        - **index** (Tensor) - 二面角的原子索引。
        - **force_constant** (Tensor) - 键扭角的谐和力常数。
        - **periodicity** (Tensor) - 扭转位垒的周期性。
        - **phase** (Tensor) - 扭转函数中的相移。
        - **parameters** (dict) - 力场参数。
        - **use_pbc** (bool) - 是否使用PBC。
        - **energy_unit** (str) - 能量单位。
        - **units** (Units) - 长度和能量单位。

    输出：
        Tensor。能量，shape为(B, 1)。

    符号：
        - **B** - Batch size。
        - **d** - 二面角的数量。
        - **D** - 模拟系统的维度。