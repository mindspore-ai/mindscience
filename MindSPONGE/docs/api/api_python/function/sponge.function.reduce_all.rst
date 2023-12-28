sponge.function.reduce_all
==============================

.. py:function:: sponge.function.reduce_all(x: Tensor, axis: Union[int, Tuple[int], List[int]] = ())

    默认情况下，通过维度中所有元素的 "逻辑与" 来减少张量的维度。
    并且还可以沿轴减小 `x` 的维度。有关详细信息请参见 `mindspore.ops.ReduceAll` 。

    参数：
        - **x** (Tensor[bool]) - 输入张量。要约简的张量的 dtype 是 bool。
          :math:`(N,*)` 其中的 :math:`*` 代表任意数量的附加维度，其维度应小于 8。
        - **axis** (Union[int, tuple(int), list(int)]) - 要减小的维度。默认值：()，减小所有维度。
          只允许常量值。必须在范围 [-rank(x), rank(x))。

    输出：
        Tensor。dtype 为 bool。
