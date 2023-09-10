sciai.utils.parse_arg
=======================

.. py:function:: sciai.utils.parse_arg(config)

    根据终端/bash输入和config字典解析参数。

    参数：
        - **config** (dict) - 配置字典。

    返回：
        Union(Namespace, object)，包含配置项的 `Namespace` 或者 `object` 。