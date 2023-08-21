mindsponge.control.Barostat
===========================

.. py:class:: mindsponge.control.Barostat(system, pressure=1.0, anisotropic=False, control_step=1, compressibility=4.6e-5, time_constant=1.0)

    压力耦合器的气压调节器。

    参数：
        - **system** (Molecule) - 模拟系统。
        - **pressure** (float) - 压力耦合参考压力P_ref (bar)。默认值：1.0。
        - **anisotropic** (bool) - 是否执行各向异性压力控制。默认值： ``False`` 。
        - **control_step** (int) - 控制器执行的步骤间隔。默认值：1。
        - **compressibility** (float) - 等温压缩率。默认值：4.6e-5。
        - **time_constant** (float) - 压力耦合的时间常数。默认值：1.0。

    输出：
        - Tensor。坐标，shape(B, A, D)。
        - Tensor。速度，shape(B, A, D)。
        - Tensor。力，shape(B, A, D)。
        - Tensor。能量，shape(B, 1)。
        - Tensor。动力学，shape(B, D)。
        - Tensor。维里，shape(B, D)。
        - Tensor。周期性边界条件box，shape(B, D)。

    .. py:method:: pressure()

        参考压力。

        返回：
            Tensor。参考压力。

    .. py:method:: compressibility()

        等温压缩率。

        返回：
            Tensor。等温压缩率。
    
    .. py:method:: pressure_scale(sim_press, ref_press, ratio=1.0)

        计算压力耦合器的坐标范围因子。

        参数：
            - **sim_press** (Tensor) - 模拟压力。
            - **ref_press** (Tensor) - 参考压力。
            - **ratio** (float) - 改变两个压力不同的比率。默认值：1.0。

        返回：
            Tensor。范围。