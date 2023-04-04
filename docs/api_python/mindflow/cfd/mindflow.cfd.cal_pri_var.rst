mindflow.cfd.cal_pri_var
=========================

.. py:function:: mindflow.cfd.cal_pri_var(con_var, material)

    从守恒量中计算原始量。

    参数：
        - **con_var** (Tensor) - 守恒量。
        - **material** (mindflow.cfd.Material) - 流体材料。

    返回：
        Tensor，shape和 `con_var` 一致。