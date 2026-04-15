sponge.colvar.BatchedPosition
==================================

.. py:class:: sponge.colvar.BatchedPosition(coordinate: Union[Tensor, Parameter, ndarray], keep_in_box: bool = False, dimension: int = 3, name: str = 'position')

    具有批处理坐标的固定位置的虚拟原子。

    参数：
        - **coordinate** (Tensor) - 张量的shape为 (a_1, a_2, ..., a_n, D) ，数据类型是float。虚拟原子的位置坐标。
        - **keep_in_box** (bool) - 是否在PBC box中替换坐标。默认值： ``False``。
        - **dimension** (int) - 仿真系统的空间维度。默认值：3。
        - **name** (str) - Colvar 的名称。默认值：'position'。