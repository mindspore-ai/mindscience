sponge.function.calc_distance
=================================

.. py:function:: sponge.function.calc_distance(position_a, position_b, pbc_box)

    计算位置A和B之间的距离。

    在分子动力学模拟中，需要通过势能对盒子(box)体积的自动微分求解压强，因此在计算势能时，必须要把体系内所有使用到的向量转变为相对盒子尺度的大小，再进行下一步的计算。
    
    若使用 pbc_box 则需要转化为同一个 pbc_box 内坐标计算 A 和B 的距离;若不使用 则用绝对坐标计算 A 和B 的距离。
    在没有周期性边界条件的情况下计算位置A和B之间的距离，用绝对坐标计算。

    参数：
        - **position_a** (Tensor) - 位置A的坐标，shape为 :math:`(B, ..., D)`。
        - **position_b** (Tensor) - 位置B的坐标，shape为 :math:`(B, ..., D)`。
        - **pbc_box** (Tensor) - 周期性盒子，shape为 :math:`(B, D)`。

    输出：
        Tensor。A和B之间的距离。shape为 :math:`(B, ..., 1)`。

    符号：
        - **B** - Batch size。
        - **D** - 模拟系统的空间维度, 一般为3。