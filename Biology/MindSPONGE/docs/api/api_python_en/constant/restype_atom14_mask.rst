restype_atom14_mask
===================

Mask of densely encoding of 21 types of amino acid (20 amino acid types and 1 unknown amino acid `UNK` ). Each row represents the corresponding amino acid. 1 in every row means the amino acid has the atom and 0 means it does not contain the atom. Shape is :math:`(21, 14)` .

.. code::

    from mindsponge.common import residue_constants
    print(residue_constants.restype_atom14_mask)