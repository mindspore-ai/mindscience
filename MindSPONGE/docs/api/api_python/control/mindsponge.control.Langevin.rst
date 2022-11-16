mindsponge.control.Langevin
===========================

.. py:class:: mindsponge.control.Langevin(system, temperature=300.0, control_step=1, time_constant=2.0, seed=0, seed2=0)

    Langevin恒温控制器。

    参考文献：
        `Goga, N.; Rzepiela, A. J.; de Vries, A. H.; Marrink, S. J.; Berendsen, H. J. C..
        Efficient Algorithms for Langevin and DPD Dynamics [J].
        Journal of Chemical Theory and Computation, 2012, 8(10): 3637-3649.
        <https://pubs.acs.org/doi/full/10.1021/ct3000876>`_。

    参数：
        - **system** (Molecule) - 模拟系统。
        - **temperature** (float) - 温度耦合的参考温度 T_ref，单位为K。默认值：300.0。
        - **control_step** (int) - 控制器执行的步骤间隔。默认值：1。
        - **time_constant** (float) - 温度耦合的时间常数 \tau_T。单位为ps。默认值：2.0。
        - **seed** (int) - 标准常态的随机种子。默认值：0。
        - **seed2** (int) - 标准常态的随机种子2。默认值：0。

    输出：
        - Tensor。坐标，shape(B, A, D)，数据类型为float。
        - Tensor。速度，shape(B, A, D)，数据类型为float。
        - Tensor。力，shape(B, A, D)，数据类型为float。
        - Tensor。能量，shape(B, 1)，数据类型为float。
        - Tensor。动力学，shape(B, D)，数据类型为float。
        - Tensor。维里，shape(B, D)，数据类型为float。
        - Tensor。周期性边界条件box，shape(B, D)，数据类型为float。

    .. py:method:: set_time_step(dt)

        设置模拟单步时间。

        参数：
            - **dt** (float) - 单步时间所需时间。