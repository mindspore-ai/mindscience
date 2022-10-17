mindsponge.potential.CoulombEnergy
==================================

.. py:class:: mindsponge.potential.CoulombEnergy(atom_charge, parameters, cutoff, use_pbc, use_pme=True, alpha, nfft, exclude_index, length_unit="nm", energy_unit="kj/mol", units)

    库伦相互作用。

    .. math::

        E_ele(r_ij) = \sum_ij k_coulomb * q_i * q_j / r_ij

    参数：
        - **atom_charge** (Tensor) - 原子电荷。
        - **parameters** (dict) - 力场参数。
        - **cutoff** (float) - 中断距离。
        - **use_pbc** (bool, 可选) - 是否使用PBC。
        - **use_pme** (bool, 可选) - 是否使用粒子网格生成条件(PME)。
        - **alpha** (float) - DSF和PME库伦相互作用中的Alpha。DSF默认值：0.25，PME默认值：0.276501。
        - **nfft** (Tensor) - 由PME要求的FFT的参数。
        - **exclude_index** (Tensor) - 由PME要求的排除索引。
        - **length_unit** (str) - 位置坐标的长度单位。
        - **energy_unit** (str) - 能量单位。
        - **units** (Units) - 长度和能量单位。

    输出：
        Tensor。能量，shape为(B, 1)。

    .. py:method:: set_cutoff(cutoff)

        设置中断距离。

        参数：
            - **cutoff** (Tensor) - 中断距离。