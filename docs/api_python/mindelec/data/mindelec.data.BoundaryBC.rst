mindelec.data.BoundaryBC
========================

.. py:class:: mindelec.data.BoundaryBC(geometry)

    边界条件采样数据。

    参数：
        - **geometry** (Geometry) - 指定边界条件的几何体信息。几何体继承mindelec.geometry.Geometry基类API，可以是圆盘或者立方体等。

    异常：
        - **ValueError** - 如果几何体的sampling_config.bc为 ``None``。
