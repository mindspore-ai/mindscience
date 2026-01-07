mindscience.e3nn.o3.change_basis_real_to_complex
======================================================

.. py:function:: mindscience.e3nn.o3.change_basis_real_to_complex(l, dtype=mindspore.float32)

    将实值球谐函数基转换为其复值对应基。
    该过程构造酉矩阵 :math:`Q`，通过关系 :math:`Y = Q \cdot y` 将实基 :math:`y_{l,m}^{\text{real}}`
    映射到标准复值基 :math:`Y_{l,m}`。:math:`Q` 的列按 :math:`m = -l,\ldots,+l` 顺序排列，所得复基满足量子力学中的相位与归一化约定。

    参数：
        - **l** (int) - 球谐函数阶数。
        - **dtype** (mindspore.dtype, 可选) - { ``mindspore.float32`` , ``mindspore.float64`` }，实基的数据类型。默认：``mindspore.float32``。

    返回：
        Tensor，复值基矩阵。若 `dtype=mindspore.float32` 则为 ``complex64``，若 `dtype=mindspore.float64` 则为 ``complex128``。

