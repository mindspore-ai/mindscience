sciai.common.TrainStepCell
============================================

.. py:class:: sciai.common.TrainStepCell(network, optimizer, grad_first=False, clip_grad=False, clip_norm=1e-3)

    具有梯度下降的 `Cell` ，类似于 `nn.TrainOneStepCell` ，但可以接受多输出。

    参数：
        - **network** (Cell) - 训练网络。网络支持多输出。
        - **optimizer** (Union[Cell]) - 用于更新网络参数的优化器。
        - **grad_first** (bool) - 若为True，则只有网络的第一个输出参与梯度下降。 否则所有输出之和参与梯度下降。默认值：False。
        - **clip_grad** (bool) - 是否裁剪梯度。默认值：False。
        - **clip_norm** (Union[float, int]) - 梯度裁剪率，需为正数. 仅当 `clip_grad` 为True时生效. 默认值：1e-3。

    输入：
        - **\*inputs** (tuple[Tensor]) - 输入张量的元组，形状为 :math:`(N, \ldots)`。

    输出：
        Union(Tensor, tuple[Tensor])，若干loss的Tensor，其形状通常是 :math:`()`。

    异常：
        - **TypeError** - 如果 `network` 或 `optimizer` 的类型不正确。