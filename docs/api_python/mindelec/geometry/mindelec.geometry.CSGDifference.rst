mindelec.geometry.CSGDifference
===============================

.. py:class:: mindelec.geometry.CSGDifference(geom1, geom2, sampling_config=None)

    几何差异的CSG类。

    参数：
        - **geom1** (Geometry) - 几何体对象。
        - **geom2** (Geometry) - 要从 `geom1` 中减去的几何体对象。
        - **sampling_config** (SamplingConfig) - 采样配置。默认值： ``None``。
