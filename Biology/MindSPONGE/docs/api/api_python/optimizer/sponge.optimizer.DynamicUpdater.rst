sponge.optimizer.DynamicUpdater
===================================

.. py:class:: sponge.optimizer.DynamicUpdater(system: Molecule, integrator: Integrator, thermostat: Controller = None, barostat: Controller = None, constraint: Controller = None, controller: Controller = None, time_step: float = 1e-3, velocity: Tensor = None, weight_decay: float = 0.0,loss_scale: float = 1.0)

    用于分子动力学（MD）仿真的更新器。
    此更新器将在未来版本中被移除，请改用 :class:`sponge.optimizer.UpdaterMD`。

    参数：
        - **system** (:class:`sponge.system.Molecule`) - 仿真系统。
        - **integrator** (`sponge.control.Integrator`) - MD积分器。
        - **thermostat** (:class:`sponge.control.Controller`, 可选) - 用于温度耦合的恒温器。默认值 ``None``。
        - **barostat** (:class:`sponge.control.Controller`, 可选) - 用于压力耦合的恒压器。默认值 ``None``。
        - **constraint** (:class:`sponge.control.Controller`, 可选) - 键约束的控制器。默认值 ``None``。
        - **controller** (:class:`sponge.control.Controller`, 可选) - 其他控制器。默认值 ``None``。
        - **time_step** (float, 可选) - 时间步长。默认值 ``1e-3``。
        - **velocity** (Tensor, 可选) - 原子速度。shape为 :math:`(B, A, D)` 。这里的 :math:`B` 是仿真中的walker的数量， :math:`A` 是原子数量， :math:`D` 是仿真系统的空间维度，通常为3。数据类型为 float。默认值： ``None``。
        - **weight_decay** (float, 可选) - 权重衰减的值。默认值 ``0.0``。
        - **loss_scale** (float, 可选) - 损失缩放的值。默认值 ``1.0``。
