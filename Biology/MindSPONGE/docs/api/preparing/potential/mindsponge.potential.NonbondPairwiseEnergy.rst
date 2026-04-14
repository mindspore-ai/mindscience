mindsponge.potential.NonbondPairwiseEnergy
==========================================

.. py:class:: mindsponge.potential.NonbondPairwiseEnergy(index=None, qiqj=None, epsilon_ij=None, sigma_ij=None, r_scale=None, r6_scale=None, r12_scale=None, parameters=None, cutoff=None, use_pbc=None, length_unit="nm", energy_unit="kj/mol", units=None)

    非键原子对的能量。

    .. math::

        E_{pairs}(r_{ij}) = A_{ij}^p \times E_r(r_{ij}) + B_{ij}^p \times E_{r6}(r_{ij}) + C_{ij}^p \times E_{r12}(r_{ij})
                        = A_{ij}^p \times k_{coulomb} \times q_i \times q_j / r_{ij} -
                          B_{ij}^p \times 4 \times \epsilon_{ij} \times (\sigma_{ij} / r_{ij}) ^ 6  +
                          C_{ij}^p \times 4 \times \epsilon_{ij} \times (\sigma_{ij} / r_{ij}) ^ {12}

    参数：
        - **index** (Tensor) - 二面角的原子索引，shape(B, p, 2)。默认值："None"。
        - **qiqj** (Tensor) - 非键原子对的电荷乘积，shape(B, p)。默认值："None"。
        - **epsilon_ij** (Tensor) - 非键原子对的epsilon，shape(B, p)。默认值："None"。
        - **sigma_ij** (Tensor) - 非键原子对的sigma，shape(B, p)。默认值："None"。
        - **r_scale** (Tensor) - 在非键相互作用中 :math:`1/r \times A^p` 项的范围常数，shape(1, p)。默认值："None"。
        - **r6_scale** (Tensor) - 在非键相互作用中 :math:`r^{-6} \times B^p` 项的范围常数，shape(1, p)。默认值："None"。
        - **r12_scale** (Tensor) - 在非键相互作用中 :math:`r^-{12} \times C^p` 项的范围常数，shape(1, p)。默认值："None"。
        - **parameters** (dict) - 力场常数。默认值："None"。
        - **cutoff** (float) - 中断距离。默认值："None"。
        - **use_pbc** (bool, 可选) - 是否使用PBC。默认值："None"。
        - **length_unit** (str) - 长度单位。默认值："nm"。
        - **energy_unit** (str) - 能量单位。默认值："kj/mol"。
        - **units** (Units) - 长度和能量单位。默认值："None"。

    输出：
        Tensor。能量， shape为(B, 1)。

    符号：
        - **B** - Batch size。
        - **p** - 非键原子对的数量。

    .. py:method:: set_cutoff(cutoff)

        设置中断距离。

        参数：
            - **cutoff** (Tensor) - 中断距离。

    .. py:method:: set_pbc(use_pbc=None)

        设置是否使用周期性边界条件PBC。

        参数：
            - **use_pbc** (bool, 可选) - 是否使用PBC。默认值："None"。