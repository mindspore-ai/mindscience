mindscience.data.SamplingConfig
==================================

.. py:class:: mindscience.data.SamplingConfig(part_sampling_dict)

    全局采样配置的定义。

    参数：
        - **part_sampling_dict** (dict) - 指定各采样部位配置的字典。字典的键表示采样部位类型，可取键包括 ``"domain"``、``"BC"``、``"IC"``、``"time"``。每个值通过 :class:`mindscience.data.PartSamplingConfig` 实例进行配置。任一支持的键均可省略，未指定的采样配置将默认设置为 ``None``。
