sciai.architecture.FirstOutputCell
=====================================

.. py:class:: sciai.architecture.FirstOutputCell(backbone)

    返回指定网络第一个输出的网络。

    参数：
        - **backbone** (Callable) - 原始网络。

    输入：
        - **\*inputs** (Tensor) - 原始网络输入。

    输出：
        Union(Tensor, tuple[Tensor])，原始网络的第一个输出。