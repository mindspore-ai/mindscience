mindsponge.common.rots_from_two_vecs
====================================

.. py:function:: mindsponge.common.rots_from_two_vecs(e0_unnormalized, e1_unnormalized)

    输入两个向量 :math:`\vec a = (a_x, a_y, a_z)` 和 :math:`\vec b = (b_x, b_y, b_z)` ，计算由这两个向量所构成的x-y平面所在坐标系与原始坐标系之间的旋转矩阵。

    首先计算 :math:`\vec a` 的单位向量 :math:`\vec e_0 = \frac{\vec a}{|\vec a|}` 作为该坐标系的x轴单位向量。

    之后计算 :math:`\vec b` 在a轴上的投影长度 :math:`c = |\vec b| \cos\theta = \vec b \cdot \frac{\vec a}{|\vec a|}` 。

    那么 :math:`\vec b` 向量在a轴上的投影向量为 :math:`c\vec e_0`，与a轴垂直的向量即为 :math:`\vec e_1' = \vec b - c\vec e_0` 。

    计算 :math:`\vec e_1'` 的单位向量 :math:`\vec e_1 = \frac{\vec e_1'}{|\vec e_1'|}` ， :math:`\vec e_1` 即为该坐标系的y轴单位向量。

    最后通过计算 :math:`\vec e_1` 和 :math:`\vec e_0` 的外积得到 :math:`\vec e_2` ，即为该坐标系的z轴单位向量。

    最后返回的旋转矩阵为 :math:`(e_0_x, e_1_x, e_2_x, e_0_y, e_1_y, e_2_y, e_0_z, e_1_z, e_2_z)` 。

    参数：
        - **e0_unnormalized** (tuple) - 作为该坐标系x轴的向量，长度为3，数据类型为标量或者shape相同的Tensor。
        - **e1_unnormalized** (tuple) - 构成X-Y平面的另一个向量，长度为3，数据类型为标量或者shape相同的Tensor。

    返回：
        tuple，两个向量的旋转矩阵 :math:`(e_0_x, e_1_x, e_2_x, e_0_y, e_1_y, e_2_y, e_0_z, e_1_z, e_2_z)` ，数据类型为标量或者shape相同的Tensor。