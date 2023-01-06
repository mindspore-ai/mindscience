restype_atom37_mask
===================

21种氨基酸残基(20种氨基酸与1种未知氨基酸 `UNK` )稀疏编码的掩码，每行表示相应氨基酸，每行中每个元素为1表示该氨基酸含有该原子，反之为0，表示不含该原子，shape为 :math:`(21, 37)`。

.. code::

    from mindsponge.common import residue_constants
    print(residue_constants.restype_atom37_mask)