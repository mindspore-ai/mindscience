mindelec.data.Equation
======================

.. py:class:: mindelec.data.Equation(geometry)

    方程域的采样数据。

    参数：
        - **geometry** (Geometry) - 指定方程域的几何信息。

    异常：
        - **TypeError** - 如果 `geometry` 不是Geometry的实例。
        - **ValueError** - 如果 `geometry` 的sampling_config为None。
        - **KeyError** - 如果 `geometry` 的sampling_config.domain为None。
