mindelec.geometry.create_config_from_edict
==========================================

.. py:function:: mindelec.geometry.create_config_from_edict(edict_config)

    从dict转换为SamplingConfig。

    参数：
        - **dict_config** (dict) - 包含配置信息的dict。

    返回：
        geometry_base.SamplingConfig，采样配置。

    异常：
        - **ValueError** - 如果无法从输入dict生成part_config_dict。
