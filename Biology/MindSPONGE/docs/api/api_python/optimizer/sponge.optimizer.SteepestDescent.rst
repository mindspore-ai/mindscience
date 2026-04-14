sponge.optimizer.SteepestDescent
====================================

.. py:class:: sponge.optimizer.SteepestDescent(params: Union[List[Parameter], List[dict]], learning_rate: Union[float, int, Tensor, Iterable, LearningRateSchedule] = 1e-03, weight_decay: Union[float, int] = 0.0, loss_scale: float = 1.0, max_shift: float = None)

    实现最陡下降（梯度下降）算法。

    .. note::
        如果参数未分组，则优化器中的 `weight_decay` 将应用于名称中不含 'beta' 或 'gamma' 的网络参数。用户可以对参数进行分组以更改权重衰减策略。当参数分组时，每个组可以设置 `weight_decay`。如果没有，则将应用优化器中的 `weight_decay`。

    参数：
        - **params** (Union[list[`mindspore.Parameter`], list[dict]]) - 必须是 `mindspore.Parameter` 的列表或字典的列表。当 `params` 是字典列表时，可以解析的键包括 "params"、"lr"、"grad_centralization" 和 "order_params"。

          - **params** - 必需。当前组中的参数。值必须是 `mindspore.Parameter` 的列表。
          - **lr** - 可选。如果键中有 "lr"，将使用相应的学习率值。
            如果没有，则使用优化器中的 `learning_rate`。支持固定和动态学习率。
          - **weight_decay** - 目前不支持通过分组参数使用不同的 `weight_decay`。
          - **grad_centralization** - 可选。必须是bool。如果键中有"grad_centralization"，则使用设置的值。 如果没有，默认为 ``False``。此配置仅在卷积层中有效。
          - **order_params** - 可选。当参数分组时，通常用于维持网络中出现的参数的顺序以提高性能。值应为优化器中将遵循的参数顺序。如果键中有 `order_params`，将忽略其他键， `order_params` 的元素必须在一个 `params` 组中。

        - **learning_rate** (Union[float, int, Tensor, Iterable, LearningRateSchedule], 可选) - 学习率。默认值 ``1e-03``。

          - float: 固定学习率值。必须大于或等于0。
          - int: 固定学习率值。必须大于或等于0。它将转换为 float。
          - Tensor: 其值应该是标量或1-D向量。对于标量，将应用固定学习率。
            对于向量，学习率是动态的，第 i 步将采用第 i 个值作为学习率。
          - Iterable: 学习率是动态的。第 i 步将采用第 i 个值作为学习率。
          - `mindspore.nn.LearningRateSchedule`: 学习率是动态的。在训练过程中，优化器调用 LearningRateSchedule 实例，并以步骤作为输入来获取当前步骤的学习率。

        - **weight_decay** (Union[float, int], 可选) - 权重衰减。必须大于或等于0。如果 `weight_decay` 是 int，将转换为 float。默认值 ``0.0``。
        - **loss_scale** (float, 可选) - 损失缩放的浮点值。必须大于0。如果 `loss_scale` 输入类型为 int，将转换为 float。通常使用默认值。只有当使用 `mindspore.amp.FixedLossScaleManager` 进行训练并且 `mindspore.amp.FixedLossScaleManager` 中的 `drop_overflow_update` 设置为 ``False`` 时，此值需要与 `mindspore.amp.FixedLossScaleManager` 中的 `loss_scale` 相同。有关更多详细信息，请参阅类 `mindspore.amp.FixedLossScaleManager`。默认值： ``1.0``。
        - **max_shift** (float, 可选) - 最大偏移的浮点值。必须大于0。它是优化器中每次迭代的偏移距离上限。如果最大偏移设置为 ``None``，则不会对偏移进行任何操作。但如果 max_shift 是一个给定的float，则偏移的界限为：[-max_shift, max_shift] 默认值： ``None``。

    输入：
        - **gradients** (Tensor) - 参数的梯度。

    输出：
        - **success** (bool) - 操作是否成功。

    异常：
        - **TypeError** - 如果 `learning_rate` 不是 int、float、Tensor、Iterable、LearningRateSchedule 中的一种。
        - **TypeError** - 如果 `parameters` 的元素既不是 Parameter 也不是 dict。
        - **TypeError** - 如果 `loss_scale` 不是 float。
        - **TypeError** - 如果 `weight_decay` 既不是 float 也不是 int。
        - **ValueError** - 如果 `loss_scale` 小于或等于 0。
        - **ValueError** - 如果 `weight_decay` 小于 0。
        - **ValueError** - 如果 `learning_rate` 是一个 Tensor，但是 tensor 的维度大于 1。
