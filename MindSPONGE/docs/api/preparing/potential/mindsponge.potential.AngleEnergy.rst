mindsponge.potential.AngleEnergy
================================

.. py:class:: mindsponge.potential.AngleEnergy(index=None, force_constant=None, bond_angle=None, parameters=None, use_pbc=None, energy_unit="kj/mol", units=None)

    键角的能量。

    .. math::

        E_{angle}(\theta_{ijk}) = 1 / 2 \times k_{ijk}^\theta \times (\theta_{ijk} - \theta_{ijk}^0) ^ 2
        
    参数：
        - **index** (Tensor) - 键角的原子索引，数据类型为int，shape为(B, a, 3)。默认值："None"。
        - **force_constant** (Tensor) - 角 :math:`(k^{\theta})` 的谐和力常数，数据类型为float，shape为(1, a)。默认值："None"。
        - **bond_angle** (Tensor) - 键角 :math:`({\theta}^0)` 的平衡值，数据类型为float，shape为(1, a)。默认值："None"。
        - **parameters** (dict) - 力场参数。默认值："None"。
        - **use_pbc** (bool) - 是否使用PBC。默认值："None"。
        - **energy_unit** (str) - 能量单位。默认值："kj/mol"。
        - **units** (Units) - 长度和能量单位。默认值："None"。

    输出：
        Tensor。能量，shape为(B, 1)。

    符号：
        - **B** - Batch size。
        - **a** - 角的数量。
        - **D** - 模拟系统的维度，一般为3。