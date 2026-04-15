restype_atom37_rigid_group_positions
====================================

Sparse encoding 21 amino acid types. Each amino acid contains the coordinates of the atoms in the local coordinate system of the rigid group to which it belongs. Shape is :math:`(21, 37, 3)` .

The position of every atom in the local coordinate system of the rigid group to which it belongs is obtained from `rigid_group_atom_positions` .

.. code::

    from mindsponge.common import residue_constants
    print(residue_constants.restype_atom37_rigid_group_positions)