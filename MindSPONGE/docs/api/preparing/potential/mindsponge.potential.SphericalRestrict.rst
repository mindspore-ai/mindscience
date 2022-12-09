mindsponge.potential.SphericalRestrict
======================================

.. py:class:: mindsponge.potential.SphericalRestrict(radius, center=0, force_constant=Energy(500, "kj/mol"), depth=Length(0.01, "nm"), length_unit=None, energy_unit=None, units=global_units, use_pbc=None)

    偏置势的基础单元。

    .. math::

        V(R) = k * log(1 + exp((|R - R_0| - r_0) / \sigma))

    参数：
        - **radius** (float) - 球体的半径。
        - **center** (Tensor) - 球心坐标。默认值：0。
        - **force_constant** (float) - 偏置势的力常数。默认值：Energy(500, 'kj/mol')
        - **depth** (float) - 壁深的限制。默认值：Length(0.01, 'nm')
        - **length_unit** (str) - 位置坐标的长度单位。默认值："None"。
        - **energy_unit** (str) - 能量单位。默认值："None"。
        - **units** (Units) - 长度和能量的单位。默认值：global_units。
        - **use_pbc** (bool) - 是否使用PBC。默认值："None"。

    输出：
        Tensor。势。