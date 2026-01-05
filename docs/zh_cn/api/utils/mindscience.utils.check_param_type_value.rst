mindscience.utils.check_param_type_value
==========================================

.. py:function:: mindscience.utils.check_param_type_value(param, param_name, valid_value, data_type=None, exclude_type=None)

    同时检查参数的数据类型与取值。

    参数：
        - **param** (any) - 待检查的参数。
        - **param_name** (str) - 参数名称（用于错误提示）。
        - **valid_value** (Union[any, tuple, list, None], 可选) - 允许的取值集合，默认 ``None``。
        - **data_type** (Union[type, tuple[type], list[type], None], 可选) - 允许的类型，默认 ``None``。
        - **exclude_type** (Union[type, tuple[type], list[type], None], 可选) - 禁止的类型，默认 ``None``。

    异常：
        - **TypeError** - 当类型检查失败时抛出。
        - **ValueError** - 当取值检查失败时抛出。