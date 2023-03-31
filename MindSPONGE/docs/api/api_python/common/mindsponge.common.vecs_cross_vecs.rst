mindsponge.common.vecs_cross_vecs
=================================

.. py:function:: mindsponge.common.vecs_cross_vecs(v1, v2)

    计算向量 :math:`v_1 = (x_1, x_2, x_3)` 和向量 :math:`v_2 = (y_1, y_2, y_3)` 的外积。

    .. math::

        cross_res = (x_2 * y_3 - x_3 * y_2, x_3 * y_1 - x_1 * y_3, x_1 * y_2 - x_2 * y_1)

    参数：
        - **v1** (tuple) - 向量 :math:`\vec v_1` ，长度为3，数据类型为标量或者shape相同的Tensor。
        - **v2** (tuple) - 向量 :math:`\vec v_2` ，长度为3，数据类型为标量或者shape相同的Tensor。

    返回：
        tuple。两个向量外积的结果，长度为3，数据类型为标量或者shape相同的Tensor。