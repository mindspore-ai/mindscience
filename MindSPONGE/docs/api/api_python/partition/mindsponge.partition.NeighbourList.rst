mindsponge.partition.NeighbourList
==================================

.. py:class:: mindsponge.partition.NeighbourList(system, cutoff=None, update_steps=20, exclude_index=None, num_neighbours=None, num_cell_cut=1, cutoff_scale=1.2, cell_cap_scale=1.25, grid_num_scale=2, large_dis=1e4, use_grids=None)

    邻居列表。

    参数：
        - **system** (Molecule) - 模拟系统。
        - **cutoff** (float) - 中断距离。默认值："None"。
        - **update_steps** (int) - 更新频率。默认值：20。
        - **exclude_index** (Tensor) - 邻居列表中可以被排除的邻居原子的索引，shape为(B, A, Ex)。默认值："None"。
        - **num_neighbours** (int) - 邻居的数量。如果输入"None"，这个值将会由邻居网格与总网格的比例来进行计算。默认值："None"。
        - **num_cell_cut** (int) - 根据中断距离划分出来的网格单元的个数。默认值：1。
        - **cutoff_scale** (float) - 缩放中断距离的因子。默认值：1.2。
        - **cell_cap_scale** (float) - 缩放容纳量的因子。默认值：1.25。
        - **grid_num_scale** (float) - 根据网格的比率计算邻居数的缩放因子。默认值：1.5。
        - **large_dis** (float) - 填充默认原子的长距离。默认值：1e4。
        - **use_grids** (bool) - 是否使用网格计算邻居列表。默认值："None"。

    符号：
        - **B** - 模拟并行线程的数量。
        - **A** - 系统中的原子数。
        - **N** - 邻居原子数。
        - **D** - 位置坐标的维度。
        - **Ex** - 被排除的邻居原子的最大数量。

    .. py:method:: calcaulate(coordinate, pbc_box=None)

        计算邻居列表。

        参数：
            - **coordinate** (Tensor) - 坐标。
            - **pbc_box** (Tensor) - PBC box。默认值："None"。

        返回：
            Tensor。索引。
            Tensor。mask。

    .. py:method:: get_neighbour_list()

        获取邻居列表。

        返回：
            Tensor。索引。
            Tensor。mask。

    .. py:method:: print_info()

        打印邻居列表的信息。

    .. py:method:: set_exclude_index(exclude_index)

        设置排除索引。

        参数：
            - **exclude_index** (Tensor) - 排除索引。

        返回：
            bool。是否成功设置。