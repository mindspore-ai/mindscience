sciai.architecture.NoArgNet
=============================

.. py:class:: sciai.architecture.NoArgNet(backbone, *inputs)

    返回指定网络第一个输出的网络。

    参数：
        - **backbone** (Cell) - 原始网络。
        - **\*inputs** (tuple) - 原始网络的输入。

    输出：
        Union(Tensor, tuple[Tensor])，指定输入经过原始网络的结果。