sponge.potential.PotentialCell
==================================

.. py:class:: sponge.potential.PotentialCell(num_energies: int = 1, energy_names: Union[str, List[str]] = 'potential', length_unit: str = None, energy_unit: str = None, use_pbc: bool = None, name: str = 'potential')

    势能的基类。

    `PotentialCell` 是 `EnergyCell` 的一个特殊子类。普通的 `EnergyCell` 只输出一个能量项，所以 `EnergyCell` 返回一个shape为 `(B, 1)` 的Tensor。
    `PotentialCell` 能够返回多个能量项，所以它的返回值是shape为 `(B, E)` 的Tensor。除此之外，默认情况下， 'PotentialCell' 的单位等于全局单位。

    参数：
        - **num_energies** (int) - 输出的能量项的数量。默认值：1
        - **energy_names** (Union[str, List[str]]) - 能量项的名字。默认值："potential"。
        - **length_unit** (str) - 长度单位。如果未被给出，则使用全局长度单位。默认值："None"。
        - **energy_unit** (str) - 能量单位。如果未被给出，则使用全局能量单位。默认值："None"。
        - **use_pbc** (bool) - 是否使用周期性边界条件。如果为None，则不使用周期性边界条件。默认值："None"。
        - **name** (str) - 能量的名字。默认值："potential"。

    输入：
        - **coordinates** (Tensor) - 系统中原子的位置坐标。shape为 (B, A, D) 的Tensor。数据类型为float。
        - **neighbour_index** (Tensor) - 相邻原子的目录。shape (B, A, N) 的Tensor。数据类型为int。默认值："None"。
        - **neighbour_mask** (Tensor) - 相邻原子的掩码。shape (B, A, N) 的Tensor。数据类型为bool。默认值："None"。
        - **neighbour_vector** (Tensor) - 从中心原子指向相邻原子的向量。shape (B, A, N, D) 的Tensor。数据类型为bool。默认值："None"。
        - **neighbour_distances** (Tensor) - 相邻原子之间的距离。shape (B, A, N) 的Tensor。数据类型为float。默认值："None"。
        - **pbc_box** (Tensor) - PBC box。shape (B, D) 的Tensor。数据类型为float。默认值："None"。

    输出：
        势，shape为 `(B, E)` 的Tensor。数据类型为float。

    .. py:method:: exclude_index()

        排除索引。

        返回：
            Tensor。排除索引。

    .. py:method:: num_energies()

        获取能量分量的数量。

        返回：
            int，能量分量的数量。

    .. py:method:: energy_names()

        获取能量名称的列表。

        返回：
            list[str]，能量名称的列表。

    .. py:method:: set_exclude_index(exclude_index)

        设置排除索引。

        参数：
            - **exclude_index** (Tensor) - 应该从非键相互作用中被排除的原子的索引。

        返回：
            Tensor，排除索引。

    .. py:method:: set_pbc(use_pbc=None)

        设置是否使用周期性边界条件PBC。

        参数：
            - **use_pbc** (bool) - 是否使用周期性边界条件。