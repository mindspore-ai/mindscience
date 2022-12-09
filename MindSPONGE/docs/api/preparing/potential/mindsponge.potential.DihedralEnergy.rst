mindsponge.potential.DihedralEnergy
===================================

.. py:class:: mindsponge.potential.DihedralEnergy(index=None, force_constant=None, periodicity=None, phase=None, parameters=None, use_pbc=None, energy_unit="kj/mol", units=None)

    二面角(扭转)的能量项。

    .. math::

        E_{dihedral}(\omega) = \sum_n 1 / 2 \times V_n \times [1 - cos(n \times \omega - {\gamma}_n)]

    参数：
        - **index** (Tensor) - 二面角的原子索引，shape为(B, d, 4)或者(1, d, 4)。默认值："None"。
        - **force_constant** (Tensor) - 键扭角的谐和力常数，shape为(B, d)或者(1, d)。默认值："None"。
        - **periodicity** (Tensor) - 扭转位垒的周期性，shape为(B, d)或者(1, d)。默认值："None"。
        - **phase** (Tensor) - 扭转函数中的相移，shape为(B, d)或者(1, d)。默认值："None"。
        - **parameters** (dict) - 力场参数。默认值："None"。
        - **use_pbc** (bool) - 是否使用PBC。默认值："None"。
        - **energy_unit** (str) - 能量单位。默认值："kj/mol"。
        - **units** (Units) - 长度和能量单位。默认值："None"。

    输出：
        Tensor。能量，shape为(B, 1)。

    符号：
        - **B** - Batch size。
        - **d** - 二面角的数量。
        - **D** - 模拟系统的维度，通常为3。