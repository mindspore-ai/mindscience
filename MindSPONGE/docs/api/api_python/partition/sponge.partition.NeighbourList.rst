sponge.partition.NeighbourList
===============================

.. py:class:: sponge.partition.NeighbourList(system: Molecule, cutoff: float = None, pace: int = 20, exclude_index: Tensor = None, num_neighbours: int = None, num_cell_cut: int = 1, cutoff_scale: float = 1.2, cell_cap_scale: float = 1.25, grid_num_scale: float = 2, use_grids: bool = False, cast_fp16: bool = False)

    近邻表。

    参数：
        - **system** (Molecule) - 模拟系统。
        - **cutoff** (float，可选) - 截断距离。如果在周期性边界条件 (PBC) 下给定了 ``None``，截断距离将被分配为默认值 1 纳米。默认值： ``None``。
        - **pace** (int，可选) - 更新近邻表的模拟频率。默认值： ``20``。
        - **exclude_index** (Tensor，可选) - 可以从近邻表中排除的相邻原子的索引Tensor，shape为 :math:`(B, A, Ex)`，数据类型为 int。默认值： ``None``。
        - **num_neighbours** (int，可选) - 最大近邻数。如果给定了 ``None``，此值将通过相邻网格数与总网格数的比例计算。默认值： ``None``。
        - **num_cell_cut** (int，可选) - 根据截断距离对网格单元进行细分的数目。默认值： ``1``。
        - **cutoff_scale** (float，可选) - 截断距离的缩放因子。默认值： ``1.2``。
        - **cell_cap_scale** (float，可选) - `cell_capacity` 的缩放因子。默认值： ``1.25``。
        - **grid_num_scale** (float，可选) - 根据网格比例计算 `num_neighbours` 的缩放因子。如果 `num_neighbours` 不是 ``None``，则不会使用该值。默认值： ``2``。
        - **use_grids** (bool，可选) - 是否使用网格来计算近邻表。默认值： ``None``。
        - **cast_fp16** (bool，可选) - 如果设置为 ``True``，数据将在排序之前转换为 float16。用于某些仅支持 float16 数据排序的设备。默认值： ``False``。

    .. note::
        - B：模拟中行走器的批量大小。
        - A：模拟系统中的原子数。
        - N：最大相邻原子数。
        - D：位置坐标的维度。
        - Ex：最大排除的近邻原子数。

    .. py:method:: calculate(coordinate: Tensor, pbc_box: Tensor = None)

        计算近邻表。

        参数：
            - **coordinate** (Tensor) - shape为 :math:`(B, A, D)` 的Tensor。数据类型为 float。位置坐标。
            - **pbc_box** (Tensor，可选) - shape为 :math:`(B, D)` 的Tensor。数据类型为 float。PBC（周期性边界条件）盒的大小。默认值： ``None``。

        返回：
            - **neigh_idx** (Tensor) - shape为 :math:`(B, A, N)` 的Tensor。数据类型为 int。系统中每个原子的相邻原子的索引。
            - **neigh_mask** (Tensor) - shape为 :math:`(B, A, N)` 的Tensor。数据类型为 bool。近邻表 `neigh_idx` 的掩码。

        .. note::
            - B：批量数，例如： 模拟中walkers的批量大小。
            - A：模拟系统中的原子数。
            - N：最大相邻原子数。
            - D：位置坐标的维度。

    .. py:method:: get_neighbour_list()

        获取近邻表。

        返回：
            - **neigh_idx** (Tensor) - Tensor，shape为 :math:`(B, A, N)`。数据类型为int。系统每个原子的邻近原子的索引。
            - **neigh_mask** (Tensor) - Tensor，shape为 :math:`(B, A, N)`。数据类型为bool。近邻表 `neigh_idx` 的掩码。

        .. note::
            - B：批量数，例如： 模拟中walkers的批量大小。
            - A：模拟系统中的原子数量。
            - N：最大邻近原子的数量。

    .. py:method:: pace
        :property:

        更新近邻表的模拟频率。

        返回：
            int, 更新后的模拟频率。

    .. py:method:: print_info()

        打印近邻表的详细信息。

    .. py:method:: set_exclude_index(exclude_index: Tensor)

        设置排除索引。

        参数：
            - **exclude_index** (Tensor) - shape为 :math:`(B, A, Ex)` 的Tensor。数据类型为 int。

    .. py:method:: update(coordinate: Tensor, pbc_box: Tensor = None)

        更新近邻表。

        参数：
            - **coordinate** (Tensor) - shape为 :math:`(B, A, D)` 的Tensor。数据类型为 float。位置坐标。
            - **pbc_box** (Tensor，可选) - shape为 :math:`(B, D)` 的Tensor。数据类型为 float。PBC（周期性边界条件）盒的大小。默认值： ``None``。

        返回：
            - **neigh_idx** (Tensor) - shape为 :math:`(B, A, N)` 的Tensor。数据类型为 int。每个原子的相邻原子的索引。
            - **neigh_mask** (Tensor) - shape为 :math:`(B, A, N)` 的Tensor。数据类型为 bool。近邻表 `neigh_idx` 的掩码。

        .. note::
            - B： 批量数，例如： 模拟中walkers的批量大小。
            - A： 模拟系统中的原子数。
            - N： 最大相邻原子数。
            - D： 位置坐标的维度。
