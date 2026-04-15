restype_rigid_group_default_frame
=================================

Affine transformation matrix of each rigid group of 21 amino acids, that is, the coordinate transformation matrix from the local coordinate system of each rigid group to the local coordinate system of the previous adjacent rigid group. Shape is :math:`(21, 8, 4, 4)` .

8 groups are
    - 0 - backbone-group Main chain rigid group corresponding to torsion angle `backbone` between atoms :math:`N-C\alpha-C-C\beta` in amino acid. CA is the origin point, C is on positive x-axis and N is on X-Y plane.
    - 1 - pre-omega-group Rigid group corresponding to torsion angle `pre-omega` between atoms :math:`N_i-C\alpha_i-N_{i-1}-C\alpha_{i-1}` in amino acid.
    - 2 - phi-group Rigid group corresponding to torsion angle `phi` between atoms :math:`C_i-C\alpha_i-N_i-C_{i+1}` in amino acid.
    - 3 - psi-group Rigid group corresponding to torsion angle `psi` between atoms :math:`N_{i-1}-C_i-C\alpha_i-N_i` in amino acid.
    - 4 - chi1-group
    - 5 - chi2-group
    - 6 - chi3-group
    - 7 - chi4-group

chi1,2,3,4-group correspond to torsion angles in the amino acids in `chi_angle_atoms` . Torsion angles are determined by the coordinates of four atoms [A, B, C, D]. Atom B and C constitute the axis of rotation, in x-axis. The third atom C is the origin point. Atom B is on the negative x-axis. The first atom A is on X-Y plane. So we can determine the coordinate of the fourth atom D.

pre-omega-group and backbone-group have the same coordinate system, and the coordinate transformation is the identity transformation.

The coordinate transformation matrix from phi-group to backbone-group, from psi-group to backbone-group, from chi1-group to backbone-group, from chi2-group to chi1-group, from chi3-group to chi2-group and from chi4-group to chi3-group need to be obtained by calculating using function `_make_rigid_transformation_4x4` .

.. code::

    from mindsponge.common import residue_constants
    print(residue_constants.restype_rigid_group_default_frame)