mindscience.data.generate_sampling_config
=============================================

.. py:function:: mindscience.data.generate_sampling_config(dict_config)

    将dict形式的采样配置转换为 `SamplingConfig` 对象。

    参数：
        - **dict_config** (dict) - 包含采样配置信息的dict。

    返回：
        geometry_base.SamplingConfig。采样配置对象。

    异常：
        - **ValueError** - 当无法从输入dict生成采样配置时抛出。
