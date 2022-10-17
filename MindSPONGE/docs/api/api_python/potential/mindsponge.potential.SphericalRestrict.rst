mindsponge.potential.SphericalRestrict
======================================

.. py:class:: mindsponge.potential.SphericalRestrict(radius, center=0, force_constant=Energy(500, "kj/mol"), depth=Length(0.01, "nm"), length_unit, energy_unit, units, use_pbc)

    偏置势的基础单元。

    .. math::

        V(R) = k * log(1 + exp((|R - R_0| - r_0) / \sigma))

    参数：
        - **radius** (float) - 球体的半径。
        - **center** (Tensor) - 球心坐标。
        - **force_constant** (float) - 偏置势的力常数。默认值：Energy(500, 'kj/mol')
        - **depth** (float) - 壁深的限制。默认值：Length(0.01, 'nm')
        - **length_unit** (str) - 位置坐标的长度单位。
        - **energy_unit** (str) - 能量单位。
        - **units** (Units) - 长度和能量的单位。
        - **use_pbc** (bool) - 是否使用PBC。

    输出：
        Tensor。势。

    符号：
        - **B** - Batch size。
        - **A** - 原子的数量。
        - **N** - 邻居原子的最大数量。
        - **D** - 模拟系统的维度。