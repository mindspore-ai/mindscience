restype_atom37_rigid_group_positions
====================================

21种氨基酸按照稀疏编码方式，每种氨基酸中所包含原子在其所属的刚体变换组的局部坐标系中的坐标。shape为 :math:`(21, 37, 3)` 。

每个原子所属的刚体变换组的局部坐标系中的坐标从 `rigid_group_atom_positions` 中获取。

.. code::

    from mindsponge.common import residue_constants
    print(residue_constants.restype_atom37_rigid_group_positions)