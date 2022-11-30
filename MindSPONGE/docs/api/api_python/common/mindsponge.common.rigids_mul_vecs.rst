mindsponge.common.rigids_mul_vecs
=================================

.. py:function:: mindsponge.common.rigids_mul_vecs(rigids, v)

    把向量 :math:`\vec v` 旋转平移到刚体变换的局部坐标系中。

    首先使用刚体变换的旋转矩阵对向量 :math:`\vec v` 进行旋转，再与平移距离相加，所得向量即为变换后的向量。

    .. math::
        v = r_rv+r_t

    参数：
        - **rigids** (tuple) - 刚体变换的旋转矩阵和平移距离。
        - **v** (tuple) - 向量 :math:`\vec v` ，长度为3，数据类型为标量或者shape相同的Tensor。

    返回：
        tuple，变换后的向量，长度为3，数据类型为标量或者shape相同的Tensor。