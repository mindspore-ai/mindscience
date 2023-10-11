sponge.function.calc_angle_by_vectors
==============================================

.. py:function:: sponge.function.calc_angle_by_vectors(vector1, vector2, keep_dims: bool = False)

    计算两个向量之间的夹角。对于向量 :math:`\vec {V_1} = (x_1, x_2, x_3, ..., x_n)` 和向量 :math:`\vec {V_2} = (y_1, y_2, y_3, ..., y_n)` ，两向量间夹角计算公式为：

    .. math::

        \theta = \arccos {\frac{|x_1y_1 + x_2y_2 + \cdots + x_ny_n|}{\sqrt{x_1^2 + x_2^2 + \cdots + x_n^2}\sqrt{y_1^2 + y_2^2 + \cdots + y_n^2}}}

    参数：
        - **vector1** (Tensor) - 向量1，shape为 :math:`(..., D)` ，数据类型为float。其中 D 为模拟系统的维度，一般为3。
        - **vector2** (Tensor) - 向量2，shape为 :math:`(..., D)` ，数据类型为float。
        - **keepdims** (bool) - 如果设置为True，则在结果中，最后一个轴将保留为大小为1的维度。默认值： ``False`` 。

    返回：
        Tensor。计算所得角。shape为 :math:`(..., 1)` ，数据类型为float。