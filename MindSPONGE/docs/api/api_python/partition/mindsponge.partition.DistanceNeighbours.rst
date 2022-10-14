mindsponge.partition.DistanceNeighbours
=======================================

.. py:class:: mindsponge.partition.DistanceNeighbours(cutoff, num_neighbours, atom_mask, exclude_index, use_pbc, cutoff_scale=1.2, large_dis=1e4)

    根据距离计算的邻居列表。

    参数：
        - **cutoff** (float) - 中断距离。
        - **num_neighbours** (int) - 邻居的数量。如果输入"None"，这个值将会由邻居网格与总网格的比例来进行计算。
        - **atom_mask** (Tensor) - 系统中原子的mask。
        - **exclude_index** (Tensor) - 邻居列表中可以被排除的邻居原子的索引。
        - **use_pbc** (bool) - 是否使用PBC。
        - **cutoff_scale** (float) - 缩放中断距离的因子。默认值：1.2。
        - **large_dis** (float) - 填充默认原子的长距离。默认值：1e4。

    符号：
        - **B** - 模拟并行线程的数量。
        - **A** - 系统中的原子数。
        - **N** - 邻居原子数。
        - **Ex** - 被排除的邻居原子的最大数量。

    .. py:method:: check_neighbours_number(neighbour_mask)

        检查邻居列表中邻居的数量。

        参数：
            - **neighbour_mask** (Tensor) - 邻居列表mask。