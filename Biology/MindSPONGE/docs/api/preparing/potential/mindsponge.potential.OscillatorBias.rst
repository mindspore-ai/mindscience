mindsponge.potential.OscillatorBias
===================================

.. py:class:: mindsponge.potential.OscillatorBias(old_crd, k, nonh_mask)

    在分子中给重原子添加限制。

    参数：
        - **old_crd** (Tensor) - 所有原子的初始坐标。
        - **k** (float) - 所有原子的弹性系数，假设相同。
        - **nonh_mask** (Tensor) - 用来区分氢原子和重原子的mask。

    输出：
        Tensor。势。