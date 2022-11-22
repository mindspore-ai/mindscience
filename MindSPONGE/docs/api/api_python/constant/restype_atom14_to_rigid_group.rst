restype_atom14_to_rigid_group
=============================

21种氨基酸按照稠密编码方式，每种氨基酸中所包含原子所属的刚体变换群，shape为 :math:`(21, 14)`。

.. code::

    from mindsponge.common import residue_constants
    print(residue_constants.restype_atom14_to_rigid_group)