mindscience.utils.check_dict_type
==================================

.. py:function:: mindscience.utils.check_dict_type(param_dict, param_name, key_type=None, value_type=None)

    检查指定字典的 key 与 value 的数据类型。

    参数：
        - **param_dict** (dict) - 待检查的字典。
        - **param_name** (str) - 参数名称（用于错误提示）。
        - **key_type** (Union[type, tuple[type], list[type], None], 可选) - 允许的 key 类型，默认 ``None``。
        - **value_type** (Union[type, tuple[type], list[type], None], 可选) - 允许的 value 类型，默认 ``None``。
