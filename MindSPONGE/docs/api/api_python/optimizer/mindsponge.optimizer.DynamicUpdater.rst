.. py:class:: mindsponge.optimizer.DynamicUpdater(system, integrator, thermostat, barostat, constraint, controller, time_step=1e-3, velocity, weight_decay=0.0, loss_scale=1.0)

    分子动力学模拟更新器。

    参数：
        - **system** (Molecule) - 模拟系统。
        - **integrator** (Integrator) - 分子动力学积分器。
        - **thermostat** (Controller) - 温度耦合恒温器。
        - **barostat** (Controller) - 压力耦合气压调节器。
        - **constraint** (Controller) - 边约束。
        - **controller** (Controller) - 其他控制器。
        - **time_step** (float) - 单步时间。默认值：1e-3。
        - **velocity** (Tensor) - 速度。
        - **weight_decay** (float) - 权重衰减值。默认值：0.0。
        - **loss_scale** (float) - 误差比例。默认值：1.0。

    输出：
        bool。更新系统的参数。

    符号：
        - **B** - Batch size。
        - **A** - 原子总数。
        - **D** - 模拟系统的维度。