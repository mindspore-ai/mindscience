mindsponge.potential.AngleEnergy
================================

.. py:class:: mindsponge.potential.AngleEnergy(index, force_constant, bond_angle, parameters, use_pbc, energy_unit="kj/mol", units)

    键角的能量。

    .. math::

        E_{angle}(\theta_{ijk}) = 1 / 2 \times k_{ijk}^\theta \times (\theta_{ijk} - \theta_{ijk}^0) ^ 2
        
    参数：
        - **index** (Tensor) - 键角的原子索引。
        - **force_constant** (Tensor) - 角的谐和力常数。
        - **bond_angle** (Tensor) - 键角的平衡值。
        - **parameters** (dict) - 力场参数。
        - **use_pbc** (bool) - 是否使用PBC。
        - **energy_unit** (str) - 能量单位。默认值："kj/mol"。
        - **units** (Units) - 长度和能量单位。

    输出：
        Tensor。能量，shape为(B, 1)。

    符号：
        - **B** - Batch size。
        - **a** - 角的数量。
        - **D** - 模拟系统的维度。