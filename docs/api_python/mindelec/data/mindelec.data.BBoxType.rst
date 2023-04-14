mindelec.data.BBoxType
======================

.. py:class:: mindelec.data.BBoxType

    采样空间边界框，仅支持立方体形状采样空间，目前支持 ``STATIC(0)`` 和 ``DYNAMIC(1)``。

    - ``'DYNAMIC'``，从所有三维拓扑模型和空间扩展的bbox生成采样bbox常量。模型bbox可以在读取所有文件后自动计算，然后在动态采样bbox可以获得的每一个方向上添加扩展名常数。每种模型都不一样。

    .. math::
        \text{Space} = (x_{min} - x_{neg}, y_{min} - y_{neg}, z_{min} - z_{neg}, x_{max} + x_{pos}, y_{max} + y_{pos}, z_{max} + z_{pos})。

    - ``'STATIC'``，用户可以指定每个维度上的采样空间，按（x_min, y_min, z_min, x_max, y_max, z_max）顺序。
