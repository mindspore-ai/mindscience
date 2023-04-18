mindelec.data.PointCloudSamplingConfig
=======================================

.. py:class:: mindelec.data.PointCloudSamplingConfig(sampling_mode, bbox_type, mode_args=None, bbox_args=None)

    PointCloud-Tensor生成的采样空间配置。

    参数：
        - **sampling_mode** (int) - 点采样方法。支持 ``0(UPPERBOUND)`` 和 ``1(DIMENSIONS)``。
        - **bbox_type** (int) - 采样空间的边界框类型，仅支持立方体形状采样空间。支持 ``0(STATIC)`` 和 ``1(DYNAMIC)``。
        - **mode_args** (Union[int, tuple]) - 采样模式的采样上界号。默认值： ``None``。
        - **bbox_args** (tuple) - 用于采样的边界参数，在不同的 `bbox_type` 中有不同的定义。默认值： ``None``。

    异常：
        - **TypeError** - 如果 `sampling_mode` 不是int。
        - **TypeError** - 如果 `bbox_type` 不是int。
        - **TypeError** - 如果 `mode_args` 不是int或tuple中的一个。
        - **TypeError** - 如果 `bbox_args` 不是tuple。
        - **TypeError** - 如果 `sampling_mode` 为 ``0``，但 `mode_args` 不是int。
        - **TypeError** - 如果 `sampling_mode` 为 ``1``，但 `mode_args` 不是三个整数的tuple。
        - **ValueError** - 如果 `sampling_mode` 为 ``1``，但 `mode_args` 的长度不是3。
        - **ValueError** - 如果 `sampling_mode` 不为 ``0(UPPERBOUND)`` 或 ``1(DIMENSIONS)``。
        - **ValueError** - 如果 `bbox_type` 不为 ``0(STATIC)`` 或 ``1(DYNAMIC)``。
