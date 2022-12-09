mindsponge.partition.GridNeighbours
===================================

.. py:class:: mindsponge.partition.GridNeighbours(cutoff, coordinate, pbc_box=None, atom_mask=None, exclude_index=None, num_neighbours=None, cell_capacity=None, num_cell_cut=1, cutoff_scale=1.2, cell_cap_scale=1.25, grid_num_scale=1.5)

    根据网格计算的邻居列表。

    参数：
        - **cutoff** (float) - 中断距离。
        - **coordinate** (Tensor) - 模拟系统中原子的位置坐标，shape为(B, A, D)。
        - **pbc_box** (Tensor) - PBC的box大小，shape为(B, A, D)。默认值："None"。
        - **atom_mask** (Tensor) - 系统中原子的mask，shape为(B, A)。默认值："None"。
        - **exclude_index** (Tensor) - 邻居列表中可以被排除的邻居原子的索引，shape为(B, A, Ex)。默认值："None"。
        - **num_neighbours** (int) - 邻居的数量。如果输入"None"，这个值将会由邻居网格与总网格的比例来进行计算。默认值："None"。
        - **cell_capacity** (int) - 网格单元中原子的容纳量。如果输入"None"，这个值将会乘以初始坐标系中的位于网格层的最大的原子数的因子。默认值："None"。
        - **num_cell_cut** (int) - 根据中断距离划分出来的网格单元的个数。默认值：1。
        - **cutoff_scale** (float) - 缩放中断距离的因子。默认值：1.2。
        - **cell_cap_scale** (float) - 缩放容纳量的因子。默认值：1.25。
        - **grid_num_scale** (float) - 根据网格的比率计算邻居数的缩放因子。默认值：1.5。

    符号：
        - **B** - 模拟并行线程的数量。
        - **A** - 系统中的原子数。
        - **D** - 位置坐标的维度。
        - **Ex** - 被排除的邻居原子的最大数量。

    .. py:method:: check_neighbours_number(grid_neigh_atoms, num_neighbours)

        检查邻居列表中邻居的数量。

        参数：
            - **grid_neigh_atoms** (Tensor) - 邻居原子的网格。
            - **num_neighbours** (int) - 邻居的数量。

    .. py:method:: get_neighbours_from_grids(atom_grid_idx, num_neighbours)

        获取网格中的邻居列表。

        参数：
            - **atom_grid_idx** (Tensor) - 原子的网格索引
            - **num_neighbours** (int) - 邻居的数量。

    .. py:method:: print_info()

        打印邻居列表的信息。

    .. py:method:: set_exclude_index(exclude_index)

        设定被排除的邻居索引。

        参数：
            - **exclude_index** (Tensor) - 被移除的邻居索引。