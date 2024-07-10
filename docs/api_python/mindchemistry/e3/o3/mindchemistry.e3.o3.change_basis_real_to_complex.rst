mindchemistry.e3.o3.change_basis_real_to_complex
======================================================

.. py:function:: mindchemistry.e3.o3.change_basis_real_to_complex(l, dtype)

    转换球谐函数实基为复基。

    参数：
        - **l** (int): 球谐函数的阶数。
        - **dtype** (dtype): {float32, float64} 实基的数据类型。默认：float32。

    返回：
        Tensor，复基，数据类型为 complex64（当 `dtype` =float32 时）和 complex128（当 `dtype` =float64 时）。

