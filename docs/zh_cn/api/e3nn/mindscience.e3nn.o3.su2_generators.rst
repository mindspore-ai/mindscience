mindscience.e3nn.o3.su2_generators
=========================================

.. py:function:: mindscience.e3nn.o3.su2_generators(j, dtype=complex64)

    计算su(2)李代数生成元。

    参数：
        - **j** (int) - 生成元的阶数。
        - **dtype** (dtype, 可选) - {``complex64``, ``complex128``}，生成器的数据类型。默认值：``complex64``。

    返回：
        Tensor，su(2) 生成元，数据类型为 ``dtype``。

    异常：
        - **TypeError** - 如果 `j` 不是整型。
