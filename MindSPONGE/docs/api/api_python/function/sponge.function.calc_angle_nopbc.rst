sponge.function.calc_angle_nopbc
==========================================

.. py:function:: sponge.function.calc_angle_nopbc(position_a, position_b, position_c, keep_dims: bool = False)

    计算非周期性边界条件下三个空间位点A，B，C所形成的角度 :math:`\angle ABC`。

    根据非周期性边界条件下A，B，C三点坐标计算向量 :math:`\vec{BA}` 和 :math:`\vec{BC}` 的坐标，再计算两向量间夹角。

    最后返回 :math:`\vec{BA}` 向量与 :math:`\vec{BC}` 向量间夹角。

    参数：
        - **position_a** (Tensor) - 位置a，shape为 :math:`(..., D)` ，数据类型为float。其中 D 为模拟系统的维度, 一般为3
        - **position_b** (Tensor) - 位置b，shape为 :math:`(..., D)` ，数据类型为float。
        - **position_c** (Tensor) - 位置c，shape为 :math:`(..., D)` ，数据类型为float。
        - **keepdims** (bool) - 如果设置为True，则在结果中，最后一个轴将保留为大小为1的维度。默认值： ``False`` 。

    输出：
        Tensor。计算所得角。shape为 :math:`(..., 1)` ，数据类型为float。