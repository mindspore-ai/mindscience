restype_atom14_mask
===================

21种氨基酸(20种氨基酸与1种未知氨基酸 `UNK` )稠密编码的掩码，每行表示相应氨基酸，每行中元素为1表示该氨基酸有该原子，反之为0，表示不含该原子，shape为 :math:`(21, 14)`。

.. code::

    from mindsponge.common import residue_constants
    print(residue_constants.restype_atom14_mask)