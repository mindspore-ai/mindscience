sponge.function.keepdims_mean
=================================

.. py:function:: sponge.function.keepdims_mean(x: Tensor, axis: Union[int, Tuple[int], List[int]] = ())

    对轴中从维度为 `x` 减小到维度为 1 的元素求平均值，并且输出和输入的维度相同。

    参数：
        - **x** (Tensor[Number]) - 输入张量。要约简的张量的 dtype 是 number。
          :math:`(N,*)` 其中的 :math:`*` 代表任意数量的附加维度，其维度应小于 8。
        - **axis** (Union[int, tuple(int), list(int)]) - 要减小的维度。默认值：()，减小所有维度。
          只允许常量值。必须在范围 [-rank(`x`), rank(`x`))。

    输出：
        Tensor。和 `x` 有相同的dtype。
