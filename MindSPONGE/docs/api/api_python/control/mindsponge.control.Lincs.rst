mindsponge.control.Lincs
========================

.. py:class:: mindsponge.control.Lincs(system, bonds='h-bonds', potential)

    LINCS 约束控制器。

    参数：
        - **system** (Molecule) - 模拟系统。
        - **bonds** (Tensor) - 需要优化的所有边。
        - **potential** (PotentialCell) - 系统的势能函数。

    输出：
        - Tensor。坐标，shape(B, A, D)。
        - Tensor。速度，shape(B, A, D)。
        - Tensor。力，shape(B, A, D)。
        - Tensor。能量，shape(B, 1)。
        - Tensor。动力学，shape(B, D)。
        - Tensor。维里，shape(B, D)。
        - Tensor。周期性边界条件box，shape(B, D)。