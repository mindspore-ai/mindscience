mindsponge.core.Sponge
======================

.. py:class:: mindsponge.core.Sponge(network, potential, optimizer, metrics, analyse_network)

    MindSPONGE的核心引擎。

    参数：
        - **network** (Cell) - 模拟系统的公式或者神经网络。
        - **potential** (Cell) - 势能函数。
        - **optimizer** (Optimizer) - 优化器。
        - **metrics** (Metric) - 矩阵。
        - **analyse_network** (Cell) - 分析网络。

    .. py:method:: analyse(dataset， callbacks)

        计算API，其中迭代由python前端控制。配置为pynative模式或CPU，计算过程将以数据集非下沉模式执行。

        参数：
            - **callbacks** (Callback) - 回调函数。
            - **dataset** (Dataset) - 评估模型的数据集。

        返回：
            Dict。key是用户定义的矩阵名称。value是测试模式中的模型的矩阵。

    .. py:method:: change_optimizer(optimizer)

        改变优化器。

        参数：
            - **optimizer** (Optimizer) - 使用的优化器。

    .. py:method:: change_potential(potential)

        改变势能。

        参数：
            - **potential** (PotentialCell) - 使用的势能。

    .. py:method:: energy()

        获取系统的能量。

        返回：
            系统的能量。

    .. py:method:: energy_and_force()

        获取能量和力。

        返回：
            系统的能量和力。

    .. py:method:: run(steps, callbacks, dataset)

        运行模拟。

        参数：
            - **steps** (int) - 模拟步数。
            - **callbacks** (Callback) - 回调函数。
            - **dataset** (Dataset) - 在模拟过程中的数据集。