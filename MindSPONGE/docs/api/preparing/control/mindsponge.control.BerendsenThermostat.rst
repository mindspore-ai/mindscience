mindsponge.control.BerendsenThermostat
======================================

.. py:class:: mindsponge.control.BerendsenThermostat(system, temperature=300.0, control_step=1, time_constant=4.0, scale_min=0.8, scale_max=1.25)

    Berendsen(弱耦合)恒温调节器。

    参考文献：
        `Berendsen, H. J. C.; Postma, J. P. M.; van Gunsteren, W. F.; DiNola, A.; Haak, J. R..
        Molecular Dynamics with Coupling to an External Bath [J].
        The Journal of Chemical Physics, 1984, 81(8): 3684.
        <https://pure.rug.nl/ws/portalfiles/portal/64380902/1.448118>`_。

    参数：
        - **system** (Molecule) - 模拟系统。
        - **temperature** (float) - 温度耦合参考压力P_ref (bar)。默认值：300.0。
        - **control_step** (int) - 控制器执行的步骤间隔。默认值：1。
        - **time_constant** (float) - 温度耦合的时间常数。默认值：4.0。
        - **scale_min** (float) - 剪裁速度范围因子的最小值。默认值：0.8。
        - **scale_max** (float) - 剪裁速度范围因子的最大值。默认值：1.25。

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