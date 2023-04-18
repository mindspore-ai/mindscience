mindflow.geometry.CSGXOR
========================

.. py:class:: mindflow.geometry.CSGXOR(geom1, geom2, sampling_config=None)

    用于几何异或计算的CSG类。

    参数：
        - **geom1** (Geometry) - 几何体对象。
        - **geom2** (Geometry) - 要与 `geom1` 求异或的几何体对象。
        - **sampling_config** (SamplingConfig) - 采样配置。默认值： ``None``。
