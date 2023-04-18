mindflow.geometry.SamplingConfig
================================

.. py:class:: mindflow.geometry.SamplingConfig(part_sampling_dict)

    全局采样配置定义。

    参数：
        - **part_sampling_dict** (dict) - 采样配置。

    异常：
        - **TypeError** - 如果 `part_sampling_dict` 不是dict。
        - **KeyError** - 如果 `geom_type` 不是 ``"domain"``、 ``"BC"``、 ``"IC"`` 或 ``"time"``。
        - **TypeError** - 如果"config"不是PartSamplingConfig对象。
        - **ValueError** - 如果 `part_sampling_dict` 中的 `domain.size` 既不是list也不是tuple。
        - **ValueError** - 如果 `part_sampling_dict` 中的 `ic.size` 既不是list也不是tuple。
        - **ValueError** - 如果 `part_sampling_dict` 中的 `time.size` 既不是list也不是tuple。
