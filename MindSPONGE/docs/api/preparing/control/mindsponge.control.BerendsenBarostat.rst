mindsponge.control.BerendsenBarostat
====================================

.. py:class:: mindsponge.control.BerendsenBarostat(system, pressure=1.0, anisotropic=False, control_step=1, compressibility=4.6e-5, time_constant=1.0)

    Berendsen(弱耦合)气压调节器。

    参考文献：
        `Berendsen, H. J. C.; Postma, J. P. M.; van Gunsteren, W. F.; DiNola, A.; Haak, J. R..
        Molecular Dynamics with Coupling to an External Bath [J].
        The Journal of Chemical Physics, 1984, 81(8): 3684.
        <https://aip.scitation.org/doi/abs/10.1063/1.448118>`_。

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

    .. py:method:: set_time_step(dt)

        设置模拟单步时间。

        参数：
            - **dt** (float) - 单步时间所需时间。