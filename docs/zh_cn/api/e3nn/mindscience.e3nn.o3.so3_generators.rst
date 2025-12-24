mindscience.e3nn.o3.so3_generators
=========================================

.. py:function:: mindscience.e3nn.o3.so3_generators(l, dtype=float32)

    计算so(3)李代数生成元。

    参数：
        - **l** (int) - 生成元的阶数。
        - **dtype** (dtype, 可选) - { ``float32`` , ``float64`` }，生成器的数据类型。默认值：``float32``。

    返回：
        Tensor，so(3) 生成元，数据类型为 ``dtype``。

    异常：
        - **TypeError** - 如果 `l` 不是整型。
        - **ValueError** - 如果矩阵数据不一致。