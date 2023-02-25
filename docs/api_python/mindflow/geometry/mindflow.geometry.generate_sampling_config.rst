mindflow.geometry.generate_sampling_config
==========================================

.. py:function:: mindflow.geometry.generate_sampling_config(dict_config)

    从dict转换为采样配置。

    参数：
        - **dict_config** (dict) - 包含配置信息的dict。

    返回：
        geometry_base.SamplingConfig，采样配置。

    异常：
        - **ValueError** - 如果无法从输入dict生成part_dict_config。
