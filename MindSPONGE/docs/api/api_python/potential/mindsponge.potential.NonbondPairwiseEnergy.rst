mindsponge.potential.NonbondPairwiseEnergy
==========================================

.. py:class:: mindsponge.potential.NonbondPairwiseEnergy(index, qiqj, epsilon_ij, sigma_ij, r_scale, r6_scale, r12_scale, parameters, cutoff, use_pbc, length_unit="nm", energy_unit="kj/mol", units)

    非键原子对的能量。

    .. math::

        E_pairs(r_{ij}) = A_{ij}^p * E_r(r_{ij}) + B_{ij}^p * E_r6(r_{ij}) + C_{ij}^p * E_r12(r_{ij})
                        = A_{ij}^p * k_coulomb * q_i * q_j / r_{ij} -
                          B_{ij}^p * 4 * \epsilon_{ij} * (\sigma_{ij} / r_{ij}) ^ 6  +
                          C_{ij}^p * 4 * \epsilon_{ij} * (\sigma_{ij} / r_{ij}) ^ 12

    参数：
        - **index** (Tensor) - 二面角的原子索引。
        - **qiqj** (Tensor) - 非键原子对的电荷乘积。
        - **epsilon_ij** (Tensor) - 非键原子对的epsilon。
        - **sigma_ij** (Tensor) - 非键原子对的sigma。
        - **r_scale** (Tensor) - 在非键相互作用中1/r项的范围常数。
        - **r6_scale** (Tensor) - 在非键相互作用中r^-6项的范围常数。
        - **r12_scale** (Tensor) - 在非键相互作用中r^-12项的范围常数。
        - **parameters** (dict) - 力场常数。
        - **cutoff** (float) - 中断距离。
        - **use_pbc** (bool, 可选) - 是否使用PBC。
        - **length_unit** (str) - 长度单位。默认值："nm"。
        - **energy_unit** (str) - 能量单位。默认值："kj/mol"。
        - **units** (Units) - 长度和能量单位。

    输出：
        Tensor。能量， shape为(B, 1)。

    符号：
        - **B** - Batch size。
        - **p** - 非键原子对的数量。
        - **D** - 模拟系统的维度。

    .. py:method:: set_cutoff(cutoff)

        设置中断距离。

        参数：
            - **cutoff** (Tensor) - 中断距离。

    .. py:method:: set_pbc(use_pbc)

        设置是否使用周期性边界条件PBC。

        参数：
            - **use_pbc** (bool, 可选) - 是否使用PBC。