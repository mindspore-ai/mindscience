mindearth.module.Trainer
=============================

.. py:class:: mindearth.module.Trainer(config, model, loss_fn, logger=None, weather_data_source="ERA5", loss_scale=DynamicLossScaleManager())
    
    Trainer类是气象预测模型训练的基类。
    所有用户自定义的预测模型训练都应该继承Trainer类。
    Trainer类根据模型输入、损失函数和相关参数生成了datasets, optimizer, callbacks, 和solver模块。例如，如果需要训练自定义模型时，可以重写get_dataset(), get_optimizer()或其他方法来满足自定义需求，或者直接实例化Trainer类。
    然后可以使用Trainer.train()方法开始训练模型。

    参数：
        - **config** (dict) - 输入参数。例如，模型参数、数据参数、训练参数。
        - **model** (mindspore.nn.Cell) - 用于训练的网络。
        - **loss_fn** (mindspore.nn.Cell) - 损失函数。
        - **logger** (logging.RootLogger, 可选) - 训练过程中的日志模块。默认值： ``None``。
        - **weatherdata_type** (str, 可选) - 数据的类型。默认值： ``Era5Data``。
        - **loss_scale** (mindspore.amp.LossScaleManager, 可选) - 使用混合精度时，用于管理损失缩放系数的类。默认值： ``mindspore.amp.DynamicLossScaleManager()``。


    异常：
        - **TypeError** - 如果 `model` 或 `loss_fn` 不是mindspore.nn.Cell。
        - **NotImplementedError** - 如果 `get_callback` 的方法没有实现。

    .. py:method:: mindearth.module.Trainer.get_callback()

        用于定义模型的回调类。用户必须自定义重写该方法。

    .. py:method:: mindearth.module.Trainer.get_checkpoint()

        获得模型的checkpoint实例。

        返回：
            Callback，模型的checkpoint实例.

    .. py:method:: mindearth.module.Trainer.get_dataset()

        获得训练数据集和验证数据集。

        返回：
            Dataset，训练数据集。
            Dataset，验证数据集。

    .. py:method:: mindearth.module.Trainer.get_optimizer()

        获得模型训练的优化器。

        返回：
            Optimizer，模型的优化器。

    .. py:method:: mindearth.module.Trainer.get_solver()

        获得模型训练的求解器。

        返回：
            Model，模型的求解器。

    .. py:method:: mindearth.module.Trainer.train()

        执行模型训练。