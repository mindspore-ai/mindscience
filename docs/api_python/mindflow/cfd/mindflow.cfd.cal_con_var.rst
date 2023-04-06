mindflow.cfd.cal_con_var
=========================

.. py:function:: mindflow.cfd.cal_con_var(pri_var, material)

    从原始量中计算守恒量。

    参数：
        - **pri_var** (Tensor) - 原始量。
        - **material** (mindflow.cfd.Material) - 流体材料。

    返回：
        Tensor，shape和 `pri_var` 一致。