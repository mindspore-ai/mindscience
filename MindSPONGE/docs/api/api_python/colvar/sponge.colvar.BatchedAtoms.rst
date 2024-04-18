sponge.colvar.BatchedAtoms
===============================

.. py:class:: sponge.colvar.BatchedAtoms(index: Union[Tensor, ndarray, List[int]], keep_in_box: bool = False, dimension: int = 3, name: str = 'atoms')

    Atoms Cell 的批处理版本。它是 `Atoms` 的一个子类。

    `BatchedAtoms` 的 `index` shape为 (B, a_1, a_2, ... , a_n) ，而 `Atoms` Cell的 `index` shape应该是 (a_1, a_2, ... , a_n) 。原子索引的batch size `B` 应与模拟系统的batch size一致。返回的 `Atoms` 单元张量的shape是 (B, a_1, a_2, ... , a_n, D) 。其中，{a_i}表示原子单元的维度。D表示仿真系统的维度，通常为3。

    参数：
        - **index** (Union[Tensor, ndarray, List[int]]) - 特定原子的索引数组。张量的shape是 (B, a_1, a_2, ... , a_n) ，并且数据类型为int。
        - **keep_in_box** (bool) - 是否在PBC框中替换坐标。默认值： ``False``。
        - **dimension** (int) - 仿真系统的空间维度。默认值：3。
        - **name** (str) - Colvar的名字。默认值：'atoms'。