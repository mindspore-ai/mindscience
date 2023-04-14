mindflow.geometry.CSGUnion
==========================

.. py:class:: mindflow.geometry.CSGUnion(geom1, geom2, sampling_config=None)

    用于几何合并的CSG类。

    参数：
        - **geom1** (Geometry) - 几何体对象。
        - **geom2** (Geometry) - 要与 `geom1` 求并集的几何体对象。
        - **sampling_config** (SamplingConfig) - 采样配置。默认值： ``None``。
