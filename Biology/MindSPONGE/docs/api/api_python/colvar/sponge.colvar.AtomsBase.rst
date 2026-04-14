sponge.colvar.AtomsBase
============================

.. py:class:: sponge.colvar.AtomsBase(keep_in_box: bool = False, dimension: int = 3, name: str = 'atoms')

    特定原子组的基类，用作MindSPONGE中的“原子组模块”。

    `AtomsBase` 单元是 `Colvar` 的一个特殊子类。它的shape (a_1, a_2, ..., a_n, D) ，其中D是原子坐标的维度（通常为 3）。与 Colvar Cell 一样，当它需要作为shape (B, A, D) 的输入坐标，它返回具有额外维度 `B` 的张量的shape，即 (B, a_1, a_2, ... , a_n, D) 。其中，B代表批量大小，即模拟中的步行者数量。{a_i}代表特定原子的维度。

    参数：
        - **keep_in_box** (bool) - 是否在PBC box中替换坐标。默认值： ``False``。
        - **dimension** (int) - 仿真系统的空间维度。默认值：3。
        - **name** (str) - Colvar的名字。默认值：'atoms'。

    .. py:method:: coordinate_in_pbc(coordinate: Tensor, pbc_box: Tensor = None)

        置换PBC box中的坐标。

    .. py:method:: get_unit(units: Units = None)

        集合变量的返回单位。

    .. py:method:: ndim()
        :property:

        原子组的秩（维数）。

    .. py:method:: reshape(input_shape: tuple)

        重新排列原子的shape。

    .. py:method:: set_dimension(dimension: int = 3)
        
        设置模拟系统的空间维度。
    
    .. py:method:: shape()
        :property:
        
        获取原子的shape。