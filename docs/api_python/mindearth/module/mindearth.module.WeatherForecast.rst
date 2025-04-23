mindearth.module.WeatherForecast
===================================

.. py:class:: mindearth.module.WeatherForecast(model, config, logger)

    WeatherForecast类是气象预测模型推理的基类。
    所有用户自定义的预测模型推理都应该继承WeatherForecast类。
    WeatherForecast类可以在训练回调或推理通过加载模型参数后被调用。
    通过调用WeatherForecast类，模型可以根据输入模型的自定义预测方法执行推理。t_out_test表示模型前向推理的次数。

    参数：
        - **model** (mindspore.nn.Cell) - 用于训练的网络。
        - **config** (dict) - 输入参数。例如，模型参数、数据参数、训练参数。
        - **logger** (logging.RootLogger) - 训练过程中的日志模块。

    .. note::
        需要重写其中的成员函数 `forecast` 用于定义模型推理的前向过程。

    .. py:method:: mindearth.module.WeatherForecast.compute_total_rmse_acc(dataset, generator_flag)

        计算数据集的总体均方根误差（RMSE）和准确率。

        该函数遍历数据集，为每个批次计算RMSE和准确率，
        并累加结果以计算整个数据集的总体RMSE和准确率。

        参数：
            - **dataset** (Dataset) - 用于计算指标的数据集对象。
            - **generator_flag** (bool) - 一个标志，指示是否使用数据生成器。

        返回：
            - 包含数据集的总体准确率和RMSE的元组。

        异常：
            - **NotImplementedError** - 如果指定了不支持的数据源。

    .. py:method:: mindearth.module.WeatherForecast.eval(dataset, generator_flag=False)

        根据验证集数据或测试集数据执行模型推理。

        参数：
            - **dataset** (mindspore.dataset) - 模型推理数据集，包括输入值和样本值。
            - **generator_flag** (bool) - 用于向 "compute_total_rmse_acc" 方法传递一个参数。指示是否使用数据生成器。默认值： ``False``。

    .. py:method:: mindearth.module.WeatherForecast.forecast(inputs, labels=None)
        :staticmethod:

        模型的预测方法。

        参数：
            - **inputs** (Tensor) - 模型的输入数据。
            - **labels** (Tensor) - 样本真实数据。默认值： ``None``。

