mindelec.solver.Solver
======================

.. py:class:: mindelec.solver.Solver(network, optimizer, loss_fn='l2', mode='Data', train_constraints=None, test_constraints=None, train_input_map=None, test_input_map=None, mtl_weighted_cell=None, latent_vector=None, latent_reg=1e-2, metrics=None, eval_network=None, eval_indexes=None, amp_level='O0', **kwargs)

    用于训练或推理的高级API。
    `Solver` 将图层分组到具有训练和推理功能的对象中。

    参数：
        - **network** (Cell) - 训练或测试网络。
        - **optimizer** (Cell) - 用于更新权重的优化器。
        - **loss_fn** (Union(str, dict, Cell)) - 目标函数，如果 `loss_fn` 为 ``None``，网络应包含逻辑损失和梯度计算。请注意，在数据模式下不支持 `loss_fn` 的dict类型。
          默认值： ``"l2"``。
        - **mode** (str) - 模型的类型。支持[``"Data"``, ``"PINNs"``]。默认值： ``"Data"``。

          - ``"Data"``：模型是data_driven。
          - ``"PINNs"``：模型是physics_informed。

        - **train_constraints** (Constraints) - 训练数据集损失的定义。默认值： ``None``。如果 `mode` 是 ``"PINNs"``，则 `train_constraints` 不能为 ``None``。
        - **test_constraints** (Constraints) - 测试数据集损失的定义。默认值： ``None``。如果 `mode` 为 ``"PINNs"``，且需要执行 `eval` （见类中的 `train_with_eval` 和 `eval` 函数）时， `test_constraints` 不能为 ``None``。
        - **train_input_map** (dict) - 在训练时，指定相应数据集中数据的列名进入网络。key为数据集的名称，value为在相应的数据集中的数据列名进入网络。默认值： ``None``。如果模型的输入不是单个， `train_input_map` 不能为 ``None``。
        - **test_input_map** (dict) - 在执行评估时，指定相应数据集中数据的列名进入网络。key为数据集的名称，value为进入网络数据集中的列名。默认值： ``None``。如果模型的输入不是单个且需要eval，则 `test_input_map` 不能为 ``None``。
        - **mtl_weighted_cell** (Cell) - 基于多任务学习不确定性评估的损失加权算法。默认值： ``None``。
        - **latent_vector** (Parameter) - 参数的Tensor。控制方程中，用于编码变分参数的潜在向量。它将与采样数据连接在一起，作为最终网络输入。默认值： ``None``。
        - **latent_reg** (float) - 潜在向量的正则化系数。默认值： ``1e-2``。
        - **metrics** (Union[dict, set]) - 在训练和推理时，由模型评估的字典或metrics集。例如：{``'accuracy'``, ``'recall'``}。默认值： ``None``。
        - **eval_network** (Cell) - 评估网络。如果未定义，`network` 和 `loss_fn` 将包装为 `eval_network`。默认值： ``None``。注：在 ``"PINNs"`` 模式下不需要设置 `eval_network` 。
        - **eval_indexes** (list) - 定义 `eval_network` 时，如果 `eval_indexes` 为 ``None``，则 `eval_network` 将传递给metrics，否则 `eval_indexes` 必须包含三个元素：损失值、预测值和标签的位置。损失值将传递给 `Loss` metrics，预测值和标签将传递到其他metric。默认值： ``None``。
        - **amp_level** (str) - `mindspore.amp.build_train_network` 中参数 `level` 的选项，混合精确训练的级别。支持[``"O0"``, ``"O2"``, ``"O3"``, ``"auto"``]。默认值： ``"O0"``。

          - ``"O0"``：不改变。
          - ``"O2"``：将网络强制转换为float16，保持批处理规范在float32中运行，使用动态损失比例。
          - ``"O3"``：将网络强制转换为float16，带有附加属性 `keep_batchnorm_fp32=False` 。
          - ``"auto"``：设置为不同设备中的建议级别。在GPU上设置级别为 ``"O2"``，Ascend上设置级别为 ``"O3"`` 。建议的级别由导出经验选择，不能总是保持平均数值。用户应指定特殊网络的级别。

          GPU上建议使用 ``"O2"``，Ascend上建议使用 ``"O3"``。有关 `amp_level` 设置的详情可查阅 `mindspore.amp.build_train_network <https://www.mindspore.cn/docs/zh-CN/master/api_python/amp/mindspore.amp.build_train_network.html#mindspore.amp.build_train_network>`_ 。

    .. py:method:: mindelec.solver.Solver.eval(valid_dataset, callbacks=None, dataset_sink_mode=True)

        由Python前端控制迭代的评估接口。
        配置为PyNative模式或CPU，评估过程将使用数据集非下沉模式执行。

        .. note::
            如果 `dataset_sink_mode` 为 ``True``，则数据将发送到设备。如果设备是Ascend，则数据功能将逐个传输。每次数据传输的限制为256M。

        参数：
            - **valid_dataset** (Dataset) - 用于评估模型的数据集。
            - **callbacks** (Optional[list(Callback)]) - 训练过程中应执行的回调对象的列表。默认值： ``None``。
            - **dataset_sink_mode** (bool) - 确定是否通过数据集通道传递数据。默认值： ``True``。

        返回：
            Dict，其键为度量的名称，值为度量的值。

    .. py:method:: mindelec.solver.Solver.predict(*predict_data)

        根据输入计算模型预测。

        .. note::
            这是一个预编译函数。参数应与model.predict()函数相同。

        参数：
            - **predict_data** (Union[Tensor, tuple(Tensor)]) - 预测数据。

        返回：
            Tensor，预测数组。

        异常：
            - **TypeError** - 如果 `predict_data` 不是Tensor或tuple[Tensor]。

    .. py:method:: mindelec.solver.Solver.train(epoch, train_dataset, callbacks=None, dataset_sink_mode=True, sink_size=-1)

        迭代由Python前端控制的训练API。

        .. note::
            如果 `dataset_sink_mode` 为 ``True``，则数据将发送到设备。如果设备是Ascend，则数据功能将逐个传输。每次数据传输的限制为256M。
            如果 `sink_size` > 0，则数据集的每个epoch都可以无限次遍历，直到从数据集中获取到 `sink_size` 个数的元素。下一个epoch继续从上一个遍历的结束位置遍历。

        参数：
            - **epoch** (int) - 通常为每个epoch数据上的迭代总数。当 `dataset_sink_mode` 设置为true且接 `sink_size` > 0时，每个epoch接收 `sink_size` 步数，而不是迭代总数。
            - **train_dataset** (Dataset) - 训练数据集迭代器。如果没有 `loss_fn` ，将会返回具有多个数据[data1, data2, data3, ...]的tuple并传递到网络。否则返回tuple[data, label]。
            - **callbacks** (Union[list[Callback], Callback]) - 回调对象或回调对象的列表，会在训练时被执行。默认值： ``None``。
            - **dataset_sink_mode** (bool) - 确定是否通过数据集通道传递数据。配置PyNative模式或CPU，训练过程中数据集将不会被下沉。默认值： ``True``。
            - **sink_size** (int) - 控制每个下沉集中的数据量。如果 `sink_size` = -1，则接收每个epoch的完整数据集。如果 `sink_size` > 0，则每个epoch下沉 `sink_size` 的数据。如果 `dataset_sink_mode` 为False，则 `sink_size` 将失效。默认值： ``-1``。

    .. py:method:: mindelec.solver.Solver.train_with_eval(epoch, train_dataset, test_dataset, eval_interval, callbacks=None, dataset_sink_mode=True, sink_size=-1)

        迭代由Python前端控制的Train_with_eval API。

        .. note::
            - 如果 `dataset_sink_mode` 为True，则数据将发送到设备。如果设备是Ascend，则数据功能将逐个传输。每次数据传输的限制为256M。
            - 如果 `sink_size` > 0，则数据集的每个epoch都可以无限次遍历，直到从数据集中获取到 `sink_size` 个数的元素。下一个epoch继续从上一个遍历的结束位置遍历。

        参数：
            - **epoch** (int) - 通常为每个epoch数据上的迭代总数。当 `dataset_sink_mode` 设置为true且接 `sink_size` > 0时，每个epoch接收 `sink_size` 步数，而不是迭代总数。
            - **train_dataset** (Dataset) - 训练数据集迭代器。如果没有 `loss_fn` ，将会返回具有多个数据[data1, data2, data3, ...]的tuple并传递到网络。否则返回tuple[data, label]。数据和标签将分别传到网络和loss函数。
            - **test_dataset** (Dataset) - 用于评估模型的数据集。
            - **eval_interval** (int) - 指定eval间隔。
            - **callbacks** (Union[list[Callback], Callback]) - 回调对象或回调对象的列表，应在训练时被执行。默认值： ``None``。
            - **dataset_sink_mode** (bool) - 确定是否通过数据集通道传递数据。配置PyNative模式或CPU，训练过程中数据集将不会被下沉。默认值： ``True``。
            - **sink_size** (int) - 控制每个下沉集中的数据量。如果 `sink_size` = -1，则接收每个epoch的完整数据集。如果 `sink_size` > 0，则每个epoch下沉 `sink_size` 的数据。如果 `dataset_sink_mode` 为False，则 `sink_size` 将失效。默认值： ``-1``。

