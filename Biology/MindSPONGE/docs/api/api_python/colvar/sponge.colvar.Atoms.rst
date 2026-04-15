sponge.colvar.Atoms
==========================

.. py:class:: sponge.colvar.Atoms(index: Union[Tensor, ndarray, List[int]], batched: bool = False, keep_in_box: bool = False, dimension: int = 3, name: str = 'atoms')

    Atoms Cell是AtomsBase的一个子类，它使用原子索引数组进行初始化。

    在初始化时，Atoms Cell接受一个原子索引数组作为输入，这个数组可以是对所有行走器都通用的，也可以对每个行走器都有一个单独的索引。

    要设置一个通用的原子索引，将 `batched` 设置为 `False` ，其中 `index` 的shape与 `Atoms` Cell的shape相同，即 (a_1, a_2, ... , a_n) ，返回的张量的shape为 (B, a_1, a_2, ... , a_n, D) 。其中B是批量大小，即模拟中的步行者数量。{a_i}是特定原子的维度。D是仿真系统的维度。通常为3。

    要为每个行走器设置单独的原子索引，将 `batched` 设置为 `True`。在这种情况下，`index` 的shape为 (B, a_1, a_2, ... , a_n) ，而 `Atoms` Cell的shape为 (a_1, a_2, ... , a_n) 。原子索引的batch size `B` 应与模拟系统的batch size相同。`Atoms` Cell返回的张量的shape为 (B, a_1, a_2, ... , a_n, D) 。

    参数：
        - **index** (Union[Tensor, ndarray, List[int]]) - 特定原子的索引数组。张量的shape为 (a_1, a_2, ..., a_n) 或 (B, a_1, a_2, ..., a_n) ，数据类型为int。
        - **batched** (bool) - 索引的第一个维度是否为批大小。默认值： ``False``。
        - **keep_in_box** (bool) - 是否在PBC box中替换坐标。默认值： ``False``。
        - **dimension** (int) - 仿真系统的空间维度。默认值：3。
        - **name** (str) - Colvar的名称。默认值：'atoms'。

    .. py:method:: reshape(input_shape: tuple)

        重新排列原子的shape。