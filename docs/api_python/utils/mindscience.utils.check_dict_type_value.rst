mindscience.utils.check_dict_type_value
======================================

.. py:function:: mindscience.utils.check_dict_type_value(param_dict, param_name, key_type=None, value_type=None, key_value=None, value_value=None)

    同时检查指定字典的 key/value 类型与取值。

    参数：
        - **param_dict** (dict) - 待检查的字典。
        - **param_name** (str) - 参数名称（用于错误提示）。
        - **key_type** (Union[type, tuple[type], list[type], None], optional) - 允许的 key 类型，默认 ``None``。
        - **value_type** (Union[type, tuple[type], list[type], None], optional) - 允许的 value 类型，默认 ``None``。
        - **key_value** (Union[any, tuple, list, None], optional) - 允许的 key 取值集合，默认 ``None``。
        - **value_value** (Union[any, tuple, list, None], optional) - 允许的 value 取值集合，默认 ``None``。

    异常：
        - **TypeError** - 当字典类型或 key/value 类型检查失败时抛出。
        - **ValueError** - 当 key/value 取值检查失败时抛出。