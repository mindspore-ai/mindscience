mindscience.utils.check_dict_value
=================================

.. py:function:: mindscience.utils.check_dict_value(param_dict, param_name, key_value=None, value_value=None)

    检查指定字典的 key 与 value 的取值范围。

    参数：
        - **param_dict** (dict) - 待检查的字典。
        - **param_name** (str) - 参数名称（用于错误提示）。
        - **key_value** (Union[any, tuple, list, None], optional) - 允许的 key 取值集合，默认 ``None``。
        - **value_value** (Union[any, tuple, list, None], optional) - 允许的 value 取值集合，默认 ``None``。

    异常：
        - **TypeError** - 当 `param_dict` 不是 `dict` 类型时抛出。
        - **ValueError** - 当 `param_dict` 的 key 的值不在 `key_value` 中时抛出。
        - **ValueError** - 当 `param_dict` 的 value 的值不在 `value_value` 中时抛出。