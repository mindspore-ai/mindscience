mindsponge.common.rots_from_two_vecs
====================================

.. py:function:: mindsponge.common.rots_from_two_vecs(e0_unnormalized, e1_unnormalized)

    输入两个向量 :math:`\vec a` 和 :math:`\vec b` ，计算由这两个向量所构成的x-y平面所在坐标系与原始坐标系之间的旋转矩阵。

    :math:`\vec a = (a_x, a_y, a_z)` ， :math:`\vec b = (b_x, b_y, b_z)`

    首先计算 :math:`\vec a` 的单位向量 :math:`\vec e_0 = \frac{\vec a}{|\vec a|}` 作为该坐标系的x轴单位向量。

    之后计算 :math:`\vec b` 在a轴上的投影长度 :math:`c = |\vec b| \cos\theta = \vec b \cdot \frac{\vec a}{|\vec a|}` 。

    那么 :math:`\vec b` 向量在a轴上的投影向量为 :math:`c\vec e_0`，与a轴垂直的向量即为 :math:`\vec e_1' = \vec b - c\vec e_0` 。

    计算 :math:`\vec e_1'` 的单位向量 :math:`\vec e_1 = \frac{\vec e_1'}{|\vec e_1'|}` ， :math:`\vec e_1` 即为该坐标系的y轴单位向量。

    最后通过计算 :math:`\vec e_1` 和 :math:`\vec e_0` 的外积得到 :math:`\vec e_2` ，即为该坐标系的z轴单位向量。

    最后返回的旋转矩阵为 :math:`(e_{0x}, e_{1x}, e_{2x}, e_{0y}, e_{1y}, e_{2y}, e_{0z}, e_{1z}, e_{2z})` 。

    参数：
        - **e0_unnormalized** (tuple) - 作为该坐标系x轴的向量，长度为3，数据类型为标量或者shape相同的Tensor。
        - **e1_unnormalized** (tuple) - 构成X-Y平面的另一个向量，长度为3，数据类型为标量或者shape相同的Tensor。

    返回：
        tuple，两个向量的旋转矩阵 :math:`(e_{0x}, e_{1x}, e_{2x}, e_{0y}, e_{1y}, e_{2y}, e_{0z}, e_{1z}, e_{2z})` ，数据类型为标量或者shape相同的Tensor。