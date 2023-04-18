mindflow.geometry.PartSamplingConfig
====================================

.. py:class:: mindflow.geometry.PartSamplingConfig(size, random_sampling=True, sampler="uniform", random_merge=True, with_normal=False, with_sdf=False)

    部分采样配置的定义。

    参数：
        - **size** (Union[int, tuple[int], list[int]]) - 采样点数。
        - **random_sampling** (bool) - 指定是否随机采样点。默认值： ``True``。
        - **sampler** (str) - 随机采样的方法。默认值： ``"uniform"``。
        - **random_merge** (bool) - 是否随机合并不同维度的坐标。默认值： ``True``。
        - **with_normal** (bool) - 是否生成边界的法向向量。默认值： ``False``。
        - **with_sdf** (bool) - 是否返回域内点的符号距离函数结果。默认值： ``False``。

    异常：
        - **TypeError** - size不为int时。
