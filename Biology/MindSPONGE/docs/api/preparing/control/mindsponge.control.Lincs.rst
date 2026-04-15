mindsponge.control.Lincs
========================

.. py:class:: mindsponge.control.Lincs(system, bonds='h-bonds', potential)

    LINCS 约束控制器。

    参数：
        - **system** (Molecule) - 模拟系统。
        - **bonds** (Tensor) - 需要优化的所有键,shape为(B,2)。默认值："h-bonds"。这个变量可以接收一个shape为(B,2)的Tensor或者"h-bonds"这个字符串。
        - **potential** (PotentialCell) - 系统的势能函数。

    输入：
        - **coordinate** (Tensor) - 系统的坐标。
        - **velocity** (Tensor) - 系统的速度。
        - **force** (Tensor) - 系统的力。
        - **energy** (Tensor) - 系统的能量。
        - **kinetics** (Tensor) - 系统的动力学。
        - **virial** (Tensor) - 系统的维里。默认值："None"。
        - **pbc_box** (Tensor) - 系统的周期性边界条件box。默认值："None"。
        - **step** (int) - 系统的步数。默认值：0。

    输出：
        - Tensor。坐标，shape(B, A, D)。
        - Tensor。速度，shape(B, A, D)。
        - Tensor。力，shape(B, A, D)。
        - Tensor。能量，shape(B, 1)。
        - Tensor。动力学，shape(B, D)。
        - Tensor。维里，shape(B, D)。
        - Tensor。周期性边界条件box，shape(B, D)。