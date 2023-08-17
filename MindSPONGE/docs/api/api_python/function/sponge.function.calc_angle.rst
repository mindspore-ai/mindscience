sponge.function.calc_angle
==============================

.. py:function:: sponge.function.calc_angle(position_a, position_b: Tensor, position_c: Tensor, pbc_box: Tensor = None)

    计算三个空间位点A，B，C所形成的角度 :math:`\angle ABC`。

    如果提供盒子尺寸pbc_box，则按照周期性边界条件下的坐标进行计算。如果盒子尺寸pbc_box为None，则按照非周期性边界条件下的坐标进行计算。

    最后返回 :math:`\vec{BA}` 向量与 :math:`\vec{BC}` 向量间夹角。

    参数：
        - **position_a** (Tensor) - 位置a，shape为 :math:`(B, ..., D)` ，数据类型为float。
        - **position_b** (Tensor) - 位置b，shape为 :math:`(B, ..., D)` ，数据类型为float。
        - **position_c** (Tensor) - 位置c，shape为 :math:`(B, ..., D)` ，数据类型为float。
        - **pbc_box** (Tensor) - PBC box，shape为 :math:`(B, D)` ，数据类型为float。默认值："None"。

    输出：
        Tensor。计算所得角，取值范围为 :math:`(0, \pi)` 。shape为 :math:`(B, ..., 1)` ，数据类型为float。

    符号：
        - **B** - Batch size。
        - **D** - 模拟系统的维度, 一般为3。