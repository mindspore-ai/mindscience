restype_atom14_rigid_group_positions
====================================

Densely encoding 21 amino acid types. Each amino acid contains the coordinates of the atoms in the local coordinate system of the rigid group to which it belongs. Shape is :math:`(21, 14, 3)` .

The position of every atom in the local coordinate system of the rigid group to which it belongs is obtained from `rigid_group_atom_positions` .

.. code::

    from mindsponge.common import residue_constants
    print(residue_constants.restype_atom14_rigid_group_positions)