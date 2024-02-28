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

    .. py:method:: mindearth.module.WeatherForecast.eval(dataset)

        根据验证集数据或测试集数据执行模型推理。

        参数：
            - **dataset** (mindspore.dataset) - 模型推理数据集，包括输入值和样本值。

    .. py:method:: mindearth.module.WeatherForecast.forecast(inputs, labels=None)
        :staticmethod:

        模型的预测方法。

        参数：
            - **inputs** (Tensor) - 模型的输入数据。
            - **labels** (Tensor) - 样本真实数据。默认值： ``None``。

