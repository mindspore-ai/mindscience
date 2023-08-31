sponge.function.calc_torsion
================================

.. py:function:: sponge.function.calc_torsion(position_a, position_b, position_c, position_d, pbc_box, keep_dims: bool = False)

    计算由四个位置A，B，C，D形成的扭转角。

    参数：
        - **position_a** (Tensor) - 位置a，shape为 :math:`(..., D)`，数据类型为float。
        - **position_b** (Tensor) - 位置b，shape为 :math:`(..., D)`，数据类型为float。
        - **position_c** (Tensor) - 位置c，shape为 :math:`(..., D)`，数据类型为float。
        - **position_d** (Tensor) - 位置d，shape为 :math:`(..., D)`，数据类型为float。
        - **pbc_box** (Tensor) - PBC box，shape为 :math:`(D)` 或 :math:`(B, D)`，其中B表示batch size，D表示模拟系统的维度，一般为3，数据类型为float。PBC box尺寸为 :math:`\vec{L}`。默认值 ``"None"``。
        - **keepdims** (bool) - 如果被设置为 ``"True"``，则最后一个轴将作为大小为 1 的维度保留在结果中。默认值 ``"False"``。

    输出：
        Tensor。计算所得扭转角。shape为 :math:`(...)` 或者 :math:`(..., 1)` ，数据类型为float。
    
    支持平台：
        ``Ascend`` ``GPU``