mindsponge.core.Sponge
======================

.. py:class:: mindsponge.core.Sponge(network: Union[Molecule, SimulationCell, RunOneStepCell], potential: PotentialCell = None, optimizer: Optimizer = None, metrics: Metric = None, analyse_network: AnalyseCell = None)

    MindSPONGE的核心引擎。


    参数：
        - **network** (Union[Molecule, SimulationCell, RunOneStepCell]) - 模拟系统的公式或者神经网络。
        - **potential** (Cell) - 势能函数。默认值："None"。
        - **optimizer** (Optimizer) - 优化器。默认值："None"。
        - **metrics** (Metric) - 矩阵。默认值："None"。
        - **analyse_network** (Cell) - 分析网络。默认值："None"。

    .. py:method:: analyse(dataset=None, callbacks=None)

        计算API，其中迭代由python前端控制。配置为pynative模式或CPU，计算过程将以数据集非下沉模式执行。

        .. note::
            如果dataset_sink_mode是True，数据将会被传输到device侧。如果端侧为Ascend，数据迁移将会依次进行。每次数据发送的最大限制为256M。当dataset_sink_mode为True时，Callback类的epoch_end方法被调用时，step_end方法将会被执行。


        参数：
            - **callbacks** (Callback) - 回调函数。默认值："None"。
            - **dataset** (Dataset) - 评估模型的数据集。默认值："None"。

        返回：
            Dict。key是用户定义的矩阵名称。value是测试模式中的模型的矩阵。

    .. py:method:: change_optimizer(optimizer: Optimizer)

        改变优化器。

        参数：
            - **optimizer** (Optimizer) - 使用的优化器。

    .. py:method:: change_potential(potential: PotentialCell)

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

    .. py:method:: run(steps: int, callbacks: Callback = None, dataset: Dataset = None)

        运行模拟。

        参数：
            - **steps** (int) - 模拟步数。
            - **callbacks** (Callback) - 回调函数。默认值："None"。
            - **dataset** (Dataset) - 在模拟过程中的数据集。默认值："None"。