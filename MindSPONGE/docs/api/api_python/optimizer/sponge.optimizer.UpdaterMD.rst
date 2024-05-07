sponge.optimizer.UpdaterMD
============================

.. py:class:: sponge.optimizer.UpdaterMD(system: Molecule, time_step: float = 1e-3, velocity: Union[Tensor, ndarray, List[float]] = None, temperature: float = None, pressure: float = None, integrator: Union[Integrator str] = 'leap_frog', thermostat: Union[Thermostat, str] = 'berendsen', barostat: Union[Barostat, str] = 'berendsen', constraint: Union[Constraint, List[Constraint], str] = None, controller: Union[Controller, List[Controller], str] = None, weight_decay: float = 0.0, loss_scale: float = 1.0, **kwargs)

    用于分子动力学（MD）仿真的更新器，是 :class:`sponge.optimizer.Updater` 的子类。

    UpdaterMD 使用四种不同的 :class:`sponge.control.Controller` 来控制仿真过程中的不同变量。 `integrator` 用于更新原子坐标和速度， `thermostat` 用于温度耦合， `barostat` 用于压力耦合， `constraint` 用于键约束。

    参数：
        - **system** (:class:`sponge.system.Molecule`) - 仿真系统。
        - **time_step** (float, 可选) - 时间步长。默认值： ``1e-3``。
        - **velocity** (Union[Tensor, ndarray, List[float]], 可选) - 原子速度数组。数组的shape为 :math:`(A, D)` 或 :math:`(B, A, D)` 。这里的 :math:`B` 是batch size， :math:`A` 是原子数量， :math:`D` 是仿真系统的空间维度，通常为3。数据类型为float。默认值： ``None``。
        - **temperature** (float, 可选) - 耦合的参考温度。仅当 `thermostat` 设置为 `str` 类型时有效。默认值： ``None``。
        - **pressure** (float, 可选) - 温度耦合的参考压力。仅当 `barostat` 设置为 `str` 类型时有效。默认值： ``None``。
        - **integrator** (Union[`sponge.control.Integrator`, str], 可选) - 用于MD仿真的积分器。可以是 `sponge.control.Integrator` 对象或积分器名称的 `str`。默认值： ``'leap_frog'``。
        - **thermostat** (Union[`sponge.control.Thermostat`, str], 可选) - 用于温度耦合的恒温器。可以是 `sponge.control.Thermostat` 对象或恒温器名称的 `str`。默认值： ``'berendsen'``。
        - **barostat** (Union[`sponge.control.Barostat`, str], 可选) - 用于压力耦合的恒压器。可以是 `sponge.control.Barostat` 对象或恒压器名称的 `str`。默认值： ``'berendsen'``。
        - **constraint** (Union[`sponge.control.Constraint`, List[`sponge.control.Constraint`], str], 可选) - 键约束的约束控制器。可以是 `sponge.control.Constraint` 对象或约束控制器名称的 `str`。默认值： ``None``。
        - **controller** (Union[:class:`sponge.control.Controller`, List[:class:`sponge.control.Controller`], str], 可选) - 其他控制器。它将在四个特定控制器（integrator, thermostat, barostat 和 constraint）之后工作。默认值： ``None``。
        - **weight_decay** (float, 可选) - 权重衰减的值。默认值： ``0.0``。
        - **loss_scale** (float, 可选) - 损失缩放的值。默认值： ``1.0``。

    输入：
        - **energy** (Tensor) - shape为 :math:`(B, 1)` 。这里的 :math:`B` 是batch size，即仿真中的walker的数量。数据类型为float。仿真系统的总势能。
        - **force** (Tensor) - shape为 :math:`(B, A, D)` 。这里的 :math:`A` 是原子数量， :math:`D` 是仿真系统的空间维度，通常为3。数据类型为float。仿真系统每个原子的力。
        - **virial** (Tensor, 可选) - shape为 :math:`(B, D, D)` 。数据类型为float。仿真系统的应力张量。默认值： ``None``。

    输出：
        - **success** (bool) - 是否成功完成当前优化步骤并进入下一步。
    
    .. py:method:: ref_press()
        :property:

        恒压器的参考压力。

        返回：
            float， 恒压器的参考压力。

    .. py:method:: ref_temp()
        :property:

        恒温器的参考温度。

        返回：
            float， 恒温器的参考温度。

    .. py:method:: set_barostat(barostat: Barostat, pressure: float = None)

        设置恒压器。

        参数：
            - **barostat** (Barostat) - 压力联轴的恒压器。
            - **pressure** (float, optional) - 恒压器的参考压力值。默认值： ``None``。

    .. py:method:: set_constraint(constraint: Union[Constraint, List[Constraint]])

        设置约束算法。

        参数：
            - **constraint** (ConstrUnion[Constraint, List[Constraint]]aint) - 约束。

    .. py:method:: set_pressure(pressure: float)

        设置恒压器的参考压力。

        参数：
            - **pressure** (float) - 恒压器的参考压力值。

    .. py:method:: set_temperature(temperature: float)

        设置恒温器的参考温度。

        参数：
            - **temperature** (float) - 恒温器的参考温度。

        返回：
            如果self.thermostat是 ``None``，返回 ``None``；否则返回thermostat设置后的参考温度的Tensor。

    .. py:method:: set_thermostat(thermostat: Thermostat, temperature: float = None)

        设置恒温器。

        参数：
            - **thermostat** (Thermostat) - 用于温度耦合的恒温器。
            - **temperature** (float, optional) - 恒温器的参考温度。默认值： ``None`` 。
