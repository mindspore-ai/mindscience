sponge.colvar.Position
===========================

.. py:class:: sponge.colvar.Position(coordinate: Union[Tensor, Parameter, ndarray], batched: bool = False, keep_in_box: bool = False, name: str = 'position')

    固定位置的虚拟原子。

    参数：
        - **coordinate** (Union[Tensor, Parameter, ndarray]) - 特定虚拟原子的位置坐标数组。张量的shape为 (a_1, a_2, ..., a_n, D) ，数据类型是float。其中，a_{i}表示特定原子的维度。D表示仿真系统的维度。通常为3。
        - **batched** (bool) - 索引的第一个维度是否为批大小。默认值： ``False``。
        - **keep_in_box** (bool) - 是否在PBC box中替换坐标。默认值： ``False``。
        - **dimension** (int) - 仿真系统的空间维度。默认值：3。
        - **name** (str) - Colvar 的名字。默认值：'position'。

    .. py:method:: reshape(input_shape: tuple)

        重新排列原子的shape。