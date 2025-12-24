mindscience.e3nn.o3.identity_angles
=========================================

.. py:function:: mindscience.e3nn.o3.identity_angles(*shape, dtype=mindspore.float32)

    返回对应于“不旋转”的欧拉角 :math:`(\alpha, \beta, \gamma)` 的单位集合。
    对于任意请求的形状，返回的三个张量均为零。

    参数：
        - **shape** (tuple[int]) - 附加维度的形状。
        - **dtype** (mindspore.dtype, 可选) - 输入张量的类型。默认值：``mindspore.float32``。

    返回：
        tuple[Tensor]，由 :math:`\alpha` 、 :math:`\beta` 、 :math:`\gamma` 组成的三元组，每个张量形状为 `shape`。

    异常：
        - **TypeError** - 如果 'shape' 不是元组类型。
        - **TypeError** - 如果 'shape' 中的元素不是整型。

