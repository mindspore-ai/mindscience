sponge.partition.GridNeighbours
================================

.. py:class:: sponge.partition.GridNeighbours(cutoff: float, coordinate: Tensor, pbc_box: Tensor = None, atom_mask: Tensor = None, exclude_index: Tensor = None, num_neighbours: int = None, cell_capacity: int = None, num_cell_cut: int = 1, cutoff_scale: float = 1.2, cell_cap_scale: float = 1.25, grid_num_scale: float = 1.5)

    网格计算近邻表。

    参数：
        - **cutoff** (float) - 截止距离。
        - **coordinate** (Tensor) - Tensor，shape为 :math:`(B, A, D)`。数据类型为 float32。模拟系统中原子的位置坐标。
        - **pbc_box** (Tensor，可选) - Tensor，shape为 :math:`(B, D)`。数据类型为 float32。周期性边界条件的盒子大小。默认值： ``None``。
        - **atom_mask** (Tensor，可选) - Tensor，shape为 :math:`(B, A)`。数据类型为 bool。系统中原子的掩码。默认值： ``None``。
        - **exclude_index** (Tensor，可选) - Tensor，shape为 :math:`(B, A, Ex)`。数据类型为 int32。可以从近邻表中排除的邻近原子的索引。默认值： ``None``。
        - **num_neighbours** (int，可选) - 邻近原子的数量。如果给定 ``None``，则该值将根据邻近网格与总网格数的比例计算。默认值： ``None``。
        - **cell_capacity** (int，可选) - 网格单元中原子的容量。如果给定 ``None``，则该值将乘以初始坐标时网格单元中最大原子数的因子。默认值： ``None``。
        - **num_cell_cut** (int，可选) - 根据截止距离对网格单元进行细分的数量。默认值： ``1``。
        - **cutoff_scale** (float，可选) - 缩放截止距离的因子。默认值： ``1.2``。
        - **cell_cap_scale** (float，可选) - 缩放 `cell_capacity` 的因子。默认值： ``1.25``。
        - **grid_num_scale** (float，可选) - 通过网格比例计算 `num_neighbours` 的缩放因子。如果 `num_neighbours` 不为 ``None``，则不会使用。默认值： ``1.5``。

    .. note::
        - B: 模拟walkers的数量。
        - A: 系统中原子的数量。
        - N: 邻近原子的数量。
        - D: 位置坐标的维度。
        - Ex: 最多排除的邻近原子数量。

    .. py:method:: check_neighbour_list()

        检查邻近邻表中的邻近原子数量。

    .. py:method:: get_neighbours_from_grids(atom_grid_idx: Tensor, num_neighbours: int)

        从网格中获取邻近原子。

        参数：
            - **atom_grid_idx** (Tensor) - Tensor，shape为 :math:`(B, A, D)`。数据类型为int。原子在网格中的索引。
            - **num_neighbours** (int) - 邻近原子的数量。

    .. py:method:: print_info()

        打印近邻表中的信息。

    .. py:method:: set_exclude_index(exclude_index: Tensor)

        设置应从近邻表中排除的原子索引。

        参数：
            - **exclude_index** (Tensor) - Tensor，shape为 :math:`(B, A, Ex)`。数据类型为int。应从近邻表中排除的原子索引。
