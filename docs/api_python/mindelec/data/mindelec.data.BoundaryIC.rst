mindelec.data.BoundaryIC
========================

.. py:class:: mindelec.data.BoundaryIC(geometry)

    初始条件的采样数据。

    参数：
        - **geometry** (Geometry) - 指定初始条件的几何体信息。

    异常：
        - **ValueError** - 如果几何体的sampling_config.ic为 ``None``。
