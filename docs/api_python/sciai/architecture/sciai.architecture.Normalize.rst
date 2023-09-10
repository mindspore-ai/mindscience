sciai.architecture.Normalize
=====================================

.. py:class:: sciai.architecture.Normalize(lb, ub)

    使用指定的下界和上界对输入进行归一化。

    参数：
        - **lb** (Tensor) - 下界。
        - **ub** (Tensor) - 上界。

    输入：
        - **inputs** (Tensor) - 要归一化的输入Tensor。

    输出：
        Tensor，归一化投影到 [-1, 1]。