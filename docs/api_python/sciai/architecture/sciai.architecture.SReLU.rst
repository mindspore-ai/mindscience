sciai.architecture.SReLU
==========================

.. py:class:: sciai.architecture.SReLU()

    Sin整流线性单元激活函数。逐个元素地应用sin整流线性单元函数。

    输入：
        - **x** (Tensor) - SReLU的输入。

    输出：
        Tensor，shape与 `x` 一致的被激活的输出。