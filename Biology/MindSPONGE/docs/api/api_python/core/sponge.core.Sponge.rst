sponge.core.Sponge
===========================

.. py:class:: sponge.core.Sponge(network: Union[Molecule, WithEnergyCell, RunOneStepCell], potential: PotentialCell = None, optimizer: Optimizer = None, metrics: dict = None, analysis: AnalysisCell = None, **kwargs)

    MindSPONGE 的核心引擎，用于模拟和分析。

    这个 Cell 是分子系统（ :class:`sponge.system.Molecule` ），势能（ :class:`sponge.potential.PotentialCell` ）和优化器（ `mindspore.nn.Optimizer` ）在 MindSPONGE 中的三个模块的顶级封装。

    有三种方式来封装这些模块：

    1. 直接将 `system`、 `potential` 和 `optimizer` 封装到 :class:`sponge.core.Sponge` 中。

    2. 先将 `system` 和 `potential` 用 :class:`sponge.core.WithEnergyCell` 封装，然后将 :class:`sponge.core.WithEnergyCell` 和 `optimizer` 一起封装到 :class:`sponge.core.Sponge` 中。在这种情况下，可以通过调整 :class:`sponge.core.WithEnergyCell` 来实现对势能的调整，例如设置 `neighbour_list` 和 :class:`sponge.core.WithEnergyCell` 中的 `bias`。

    3. 先将 `system` 和 `potential` 用 :class:`sponge.core.WithEnergyCell` 封装，然后将 :class:`sponge.core.WithEnergyCell` 和 `optimizer` 一起封装到 :class:`sponge.core.RunOneStepCell` 中，最后将 :class:`sponge.core.RunOneStepCell` 传递到 :class:`sponge.core.Sponge` 中。在这种情况下，可以通过调整 :class:`sponge.core.RunOneStepCell` 来实现对力的调整，例如在 :class:`sponge.core.RunOneStepCell` 中添加 :class:`sponge.core.WithForceCell`。

    对于模拟：

    通过执行成员函数 :func:`sponge.core.Sponge.run` 来进行模拟。

    对于分析：

    :class:`sponge.core.Sponge` 还可以通过 `metrics` 来分析模拟系统。 `metrics` 应该是 :class:`sponge.metrics.Metric` 或 :class:`sponge.colvar.Colvar` 的字典。可以通过执行成员函数 :func:`sponge.core.Sponge.analyse` 来计算 `metrics` 的值。

    参数:
        - **network** (Union[Molecule, WithEnergyCell, RunOneStepCell]) - 模拟系统的 Cell。数据类型参考 :class:`sponge.system.Molecule` 、 :class:`sponge.core.WithEnergyCell` 和 :class:`sponge.core.RunOneStepCell` 。
        - **potential** (:class:`sponge.potential.PotentialCell`, 可选) - 势能。默认值: ``None``。
        - **optimizer** (`mindspore.nn.Optimizer`, 可选) - 优化器。默认值: ``None``。
        - **metrics** (dict, 可选) - 用于系统分析的metrics字典。字典的键类型应为 `str`，值类型应为 :class:`sponge.metrics.Metric` 或 :class:`sponge.colvar.Colvar`。默认值: ``None``。
        - **analysis** (:class:`sponge.core.AnalysisCell`, 可选) - 分析网络。默认值: ``None``。

    .. py:method:: analyse(dataset: Dataset = None, callbacks: Union[Callback, List[Callback]] = None)

        分析API。

        .. note::
            要使用此API，必须在 :class:`sponge.core.Sponge` 初始化时设置 `metrics`。

        参数：
            - **dataset** (Dataset) - 要分析的模拟数据集。默认值： ``None``。
            - **callbacks** (Union[`mindspore.train.Callback`, List[`mindspore.train.Callback`]]) - 训练期间应执行的回调对象列表。默认值： ``None``。

        返回：
            Dict，键是用户定义的度量名称，值是测试模式下模型的度量值。

    .. py:method:: calc_biases()
        
        计算模拟系统的各个偏置势项。
        
        返回：
            Tensor，shape为 :math:`(B, V)` 的Tensor。这里 :math:`B` 是batch size， :math:`V` 是偏置势项的数量。数据类型为float。偏置势项。

    .. py:method:: calc_energy()

        计算模拟系统的总势能（势能和偏置势）。

        返回：
            shape为 :math:`(B, 1)` 的Tensor。这里 :math:`B` 是batch size。数据类型为float。总势能。

    .. py:method:: calc_energies()

        计算模拟系统的各个能量项。

        返回：
            List[Tensor]， 包含各个能量项的Tensor列表。每个张量的形状为 :math:`(B, U)` ，这里 :math:`B` 是batch size, :math:`U` 是能量项的数量。数据类型为float。

    .. py:method:: calc_potential()

        计算并返回势能。

        返回：
            Tensor，shape为 :math:`(B, 1)` 的Tensor。总势能。这里 :math:`B` 是batch size，即模拟中的步行者数量。数据类型为float。

    .. py:method:: change_optimizer(optimizer: Optimizer)

        更改优化器。

        参数：
            - **optimizer** (:class:`mindsponge.optimizer.Optimizer`) - 优化器。

    .. py:method:: change_potential(potential: PotentialCell)

        更改势能。

        参数：
            - **potential** (:class:`sponge.potential.PotentialCell`) - 势能。

    .. py:method:: energy_names()
        :property:

        能量项的名称。

        返回：
            List[str]。能量项的名称。

    .. py:method:: get_bias()

        获取总偏置势能。

        返回：
            Tensor，shape为 :math:`(B, 1)` 的Tensor。这里 :math:`B` 是batch size，即模拟中的步行者数量。数据类型为float。

    .. py:method:: get_biases()

        获取偏置势。

        返回：
            Tensor，shape为 :math:`(B, V)` 的Tensor。偏置势项。
            这里 :math:`B` 是batch size，即模拟中的步行者数量，:math:`V` 是偏置势项的数量。数据类型为float。

    .. py:method:: get_energies()

        获取势能项。

        返回：
            Tensor，shape为 :math:`(B, U)` 的Tensor。势能项。这里 :math:`B` 是batch size，即模拟中的步行者数量，:math:`U` 是势能项的数量。数据类型为float。

    .. py:method:: num_biases()
        :property:

        偏置势项的数量 V。

        返回：
            int。偏置势项的数量。

    .. py:method:: num_energies()
        :property:

        能量项的数量。

        返回：
            int。能量项的数量。

    .. py:method:: recompile()

        重新编译模拟网络。

    .. py:method:: run(steps: int, callbacks: Union[Callback, List[Callback]] = None, dataset: Dataset = None, show_time: bool = True)

        运行模拟的接口函数。

        参数：
            - **steps** (int) - 步骤数。
            - **callbacks** (Union[`mindspore.train.Callback`, List[`mindspore.train.Callback`]]) - 获取模拟系统信息的回调函数。默认值: ``None``。
            - **dataset** (Dataset) - 模拟过程中使用的数据集。默认值: ``None``。
            - **show_time** (bool) - 是否显示时间。默认值: ``True``。

    .. py:method:: update_bias(step: int)

        更新偏置势。

        参数：
            - **step** (int) - 步骤数。

    .. py:method:: update_modifier(step: int)

        更新力修饰器。

        参数：
            - **step** (int) - 仿真step。

    .. py:method:: update_neighbour_list()

        更新邻居列表。

    .. py:method:: update_wrapper(step: int)

        更新能量包装器。

        参数：
            - **step** (int) - 步骤数。
