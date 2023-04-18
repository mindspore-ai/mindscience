mindelec.geometry.create_config_from_edict
==========================================

.. py:function:: mindelec.geometry.create_config_from_edict(edict_config)

    从dict转换为SamplingConfig。

    参数：
        - **edict_config** (dict) - 包含配置信息的dict。键可以为 ``"domain"``、 ``"BC"``、 ``"IC"`` 或 ``"time"``，对应每个键的值仍为dict，其中可以包含以下键名。

          - ``'size'`` - 采样点数, 值类型: Union[int, tuple[int], list[int]]。
          - ``'random_sampling'`` - 指定是否随机采样点，值类型: bool。
          - ``'sampler'`` - 随机采样的方法，值类型: str。
          - ``'random_merge'`` - 是否随机合并不同维度的坐标，值类型: bool。
          - ``'with_normal'`` - 是否生成边界的法向向量，值类型: bool。

    返回：
        geometry_base.SamplingConfig，采样配置。

    异常：
        - **ValueError** - 如果输入与GEOM_TYPES完全无法匹配，则无法从输入dict生成part_config_dict。
