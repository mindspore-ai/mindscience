mindchemistry.e3.nn.soft_unit_step
====================================

.. py:function:: mindchemistry.e3.nn.soft_unit_step(x)

    单位阶跃函数的平滑版本。

    .. math::
        x \mapsto \theta(x) e^{-1/x}

    参数：
        - **x** (Tensor) - 输入张量。

    返回：
        张量，单位阶跃函数的输出。




