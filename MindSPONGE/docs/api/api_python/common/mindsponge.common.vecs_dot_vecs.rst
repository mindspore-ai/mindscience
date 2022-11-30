mindsponge.common.vecs_dot_vecs
===============================

.. py:function:: mindsponge.common.vecs_dot_vecs(v1, v2)

    计算向量 :math:`v_1 = (x_1, x_2, x_3)` 和向量 :math:`v_2 = (y_1, y_2, y_3)` 的内积。

    .. math::

        res = x_1 * y_1 + x_2 * y_2 + x_3 * y_3

    参数：
        - **v1** (tuple) - 向量 :math:`v_1` ，长度为3，数据类型为标量或者shape相同的Tensor。
        - **v2** (tuple) - 向量 :math:`v_2` ，长度为3，数据类型为标量或者shape相同的Tensor。

    返回：
        标量或Tensor，形状与输入中的数据相同。两个向量内积的结果。