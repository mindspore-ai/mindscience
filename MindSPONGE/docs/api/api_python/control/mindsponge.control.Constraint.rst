mindsponge.control.Constraint
=============================

.. py:class:: mindsponge.control.Constraint(system, bonds='h-bonds', potential)

    边的约束。

    参数：
        - **system** (Molecule) - 模拟体系。
        - **bonds** (Tensor, str) - 被约束的边，shape(K, 2)。默认值："h-bonds"。
        - **potential** (PotentialCell) - 势能层。

    输出：
        - Tensor。坐标，shape(B, A, D)。
        - Tensor。速度，shape(B, A, D)。
        - Tensor。力，shape(B, A, D)。
        - Tensor。能量，shape(B, 1)。
        - Tensor。动力学，shape(B, D)。
        - Tensor。维里，shape(B, D)。
        - Tensor。周期性边界条件box，shape(B, D)。