mindscience.e3nn.o3.rand_angles
=====================================

.. py:function:: mindscience.e3nn.o3.rand_angles(*shape)

    返回一组均匀随机的欧拉角 :math:`(\alpha, \beta, \gamma)`，对应三维空间中的随机旋转。
    :math:`\alpha` 与 :math:`\gamma` 在区间 \([0, 2\pi)\) 上均匀采样，:math:`\beta` 在 \([0, \pi]\) 上按概率密度与 \(\sin(\beta)\) 成正比采样，以保证在旋转群 SO(3) 上的均匀分布。

    参数：
        - **shape** (tuple[int]) - 附加尺寸的形状。

    返回：
        tuple[Tensor]，由 :math:`\alpha` 、:math:`\beta` 、:math:`\gamma` 组成的三元组，每个张量形状为 `shape`。

    异常：
        - **TypeError** - 如果 `shape` 的类型不是 tuple。
        - **TypeError** - 如果 `shape` 元素的类型不是 int。