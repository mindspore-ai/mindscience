mindelec.geometry.CSGIntersection
=================================

.. py:class:: mindelec.geometry.CSGIntersection(geom1, geom2, sampling_config=None)

    几何相交的Constructive Solid Geometry(CSG)类。

    参数：
        - **geom1** (Geometry) - 几何体对象。
        - **geom2** (Geometry) - 要与geom1求交集的几何体对象。
        - **sampling_config** (SamplingConfig) - 采样配置。默认值： ``None``。
