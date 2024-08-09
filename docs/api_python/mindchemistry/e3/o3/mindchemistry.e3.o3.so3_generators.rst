mindchemistry.e3.o3.so3_generators
=========================================

.. py:function:: mindchemistry.e3.o3.so3_generators(l, dtype=float32)

    计算so(3)李代数生成元。

    参数：
        - **l** (int) - 生成维度。
        - **dtype** (dtype) - ｛float32, float64｝生成器的数据类型。默认值: ``float32``。

    返回：
        - **output** (Tensor) - 张量，计算so(3)李代数生成元结果，类型为dtype。

    异常：
        - **TypeError** - 如果 `j` 不是整型。
        - **ValueError** - 如果矩阵数据不一致。