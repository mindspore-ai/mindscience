mindsponge.potential.CoulombEnergy
==================================

.. py:class:: mindsponge.potential.CoulombEnergy(atom_charge=None, parameters=None, cutoff=None, use_pbc=None, use_pme=None, alpha=0.25, nfft=None, exclude_index=None, length_unit="nm", energy_unit="kj/mol", units=None)

    库伦相互作用。

    .. math::

        E_{ele}(r_{ij}) = \sum_{ij} k_{coulomb} \times q_i \times q_j / r_{ij}

    参数：
        - **atom_charge** (Tensor) - 原子电荷，数据类型为float。默认值："None"。
        - **parameters** (dict) - 力场参数。默认值："None"。
        - **cutoff** (float) - 中断距离。默认值："None"。
        - **use_pbc** (bool, 可选) - 是否使用PBC。默认值："None"。
        - **use_pme** (bool, 可选) - 是否使用粒子网格生成条件(PME)。默认值："None"。
        - **alpha** (float) - DSF和PME库伦相互作用中的Alpha。DSF默认值：0.25，PME默认值：0.276501。
        - **nfft** (Tensor) - 由PME要求的FFT的参数。默认值："None"。
        - **exclude_index** (Tensor) - 由PME要求的排除索引。默认值："None"。
        - **length_unit** (str) - 位置坐标的长度单位。默认值："nm"。
        - **energy_unit** (str) - 能量单位。默认值："kj/mol"。
        - **units** (Units) - 长度和能量单位。默认值："None"。

    输出：
        Tensor。能量，shape为(B, 1)，数据类型为float。

    .. py:method:: set_cutoff(cutoff)

        设置中断距离。

        参数：
            - **cutoff** (Tensor) - 中断距离。