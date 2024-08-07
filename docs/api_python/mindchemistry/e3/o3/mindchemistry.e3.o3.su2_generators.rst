mindchemistry.e3.o3.su2_generators
=========================================

.. py:function:: mindchemistry.e3.o3.su2_generators(j, dtype=complex64)

    计算su(2)李代数生成元。

    参数：
        - **j** (int) - 生成维度。
        - **dtype** (dtype) - ｛complex64，complex128｝生成器的数据类型。默认值: ``complex64``。

    返回：
        - **output** (Tensor) - 张量，计算su(2)李代数生成元结果，类型为dtype。

    异常：
        - **TypeError** - 如果 `j` 不是整型。
