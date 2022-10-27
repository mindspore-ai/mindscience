mindsponge.core.RunOneStepCell
==============================

.. py:class:: mindsponge.core.RunOneStepCell(network, optimizer, steps=1, sens=1.0)

    运行一步模拟的核心层。

    参数：
        - **network** (SimulationCell) - 模拟系统的网络。
        - **optimizer** (Optimizer) - 模拟优化器。
        - **steps** (int) - JIT的步数。默认值：1。
        - **sens** (float) - 作为反向传播的输入要填充的缩放数。默认值：1.0。

    .. py:method:: get_energy_and_force(*inputs)

        获取系统的能量和力。

        返回：
            - Tensor。能量。
            - Tensor。力。

    .. py:method:: run_one_step(*inputs)

        运行单步模拟。

        返回：
            - Tensor。模拟层结果输出的能量的大小。
            - Tensor。模拟层结果输出的力的大小。

    .. py:method:: set_pbc_grad(value)

        设定是否计算PBC box的梯度。

        参数：
            - **value** (bool) - 判断是否计算PBC box的梯度。

    .. py:method:: set_steps(steps)

        设置JIT的步数。

        参数：
            - **steps** (int) - JIT的步数。