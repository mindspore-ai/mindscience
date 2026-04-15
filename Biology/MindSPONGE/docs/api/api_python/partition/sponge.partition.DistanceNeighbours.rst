sponge.partition.DistanceNeighbours
====================================

.. py:class:: sponge.partition.DistanceNeighbours(cutoff: float, num_neighbours: int = None, atom_mask: Tensor = None, exclude_index: Tensor = None, use_pbc: bool = None, cutoff_scale: float = 1.2, large_dis: float = 1e4, cast_fp16: bool = False)

    根据距离计算近邻表。

    参数：
        - **cutoff** (float) - 截止距离。
        - **num_neighbours** (int，可选) - 邻近原子的数量。如果给定 ``None``，则该值将根据邻近网格与总网格数的比例计算。默认值： ``None``。
        - **atom_mask** (Tensor，可选) - Tensor，shape为 :math:`(B, A)`。数据类型为bool。系统中原子的掩码。默认值： ``None``。
        - **exclude_index** (Tensor，可选) - Tensor，shape为 :math:`(B, A, Ex)`。数据类型为int32。可以从近邻表中排除的邻近原子的索引。默认值： ``None``。
        - **use_pbc** (bool，可选) - 是否使用周期性边界条件。默认值： ``None``。
        - **cutoff_scale** (float，可选) - 缩放截止距离的因子。默认值： ``1.2``。
        - **large_dis** (float，可选) - 一个大数，用于填充被掩码邻近原子的距离。默认值： ``1e4``。
        - **cast_fp16** (bool，可选) - 如果设置为 ``True``，在排序前将数据转换为float16。用于一些仅支持float16数据排序的设备。默认值： ``False``。

    .. note::
        - B：模拟行者的数量。
        - A：系统中原子的数量。
        - N：邻近原子的数量。
        - Ex：最多排除的邻近原子数量。

    .. py:method:: calc_max_neighbours(distances: Tensor, cutoff: float)

        计算近邻原子的最大数量。

        参数：
            - **distances** (Tensor) - Tensor，shape为 :math:`(B, A, N)`。数据类型为float。邻近原子的距离。
            - **cutoff** (float) - 截止距离。
    
    .. py:method:: check_neighbour_list()

        检查邻近邻表中的邻近原子数量。

    .. py:method:: print_info()

        打印近邻表中的信息。

    .. py:method:: set_exclude_index(exclude_index: Tensor)

        设置应从近邻表中排除的原子索引。

        参数：
            - **exclude_index** (Tensor) - Tensor，shape为 :math:`(B, A, Ex)`。数据类型为int。应从近邻表中排除的原子索引。

    .. py:method:: set_num_neighbours(coordinate: Tensor, pbc_box: Tensor = None, scale_factor: float = 1.25)

        设置近邻原子的最大数量。

        参数：
            - **coordinate** (Tensor) - Tensor，shape为 :math:`(B, A, D)`。数据类型为float。原子的位置坐标。
            - **pbc_box** (Tensor，可选) - Tensor，shape为 :math:`(B, D)`。数据类型为bool。周期性边界条件盒子。默认值： ``None``。
            - **scale_factor** (float，可选) - 缩放系数。默认值： ``1.25``。
