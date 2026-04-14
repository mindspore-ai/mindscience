restype_atom14_to_rigid_group
=============================

21种氨基酸按照稠密编码方式，每种氨基酸中所包含原子所属的刚体变换组，shape为 :math:`(21, 14)`。

8种group为
    - 0 - backbone-group 氨基酸 :math:`N-C\alpha-C-C\beta` 原子之间的扭转角 `backbone` 对应主链刚体变换组，CA为坐标原点，C在x轴正向，N在X-Y平面。
    - 1 - pre-omega-group 氨基酸 :math:`N_i-C\alpha_i-N_{i-1}-C\alpha_{i-1}` 原子之间的扭转角 `pre-omega` 对应刚体变换组。
    - 2 - phi-group 氨基酸 :math:`C_i-C\alpha_i-N_i-C_{i+1}` 原子之间的扭转角 `phi` 对应刚体变换组。
    - 3 - psi-group 氨基酸 :math:`N_{i-1}-C_i-C\alpha_i-N_i` 原子之间的扭转角 `psi` 对应刚体变换组。
    - 4 - chi1-group
    - 5 - chi2-group
    - 6 - chi3-group
    - 7 - chi4-group

chi1,2,3,4-group 在 `chi_angle_atoms` 相应氨基酸的相应扭转角中，扭转角由四个原子[A,B,C,D]的坐标确定，中间两个原子B,C为旋转轴，在x轴上；第3个原子C为坐标原点，原子B在x负半轴；第1个原子A在xy平面，从而确定该坐标下第4个原子坐标。

.. code::

    from mindsponge.common import residue_constants
    print(residue_constants.restype_atom14_to_rigid_group)