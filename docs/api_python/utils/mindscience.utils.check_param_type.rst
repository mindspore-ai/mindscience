mindscience.utils.check_param_type
=================================

.. py:function:: mindscience.utils.check_param_type(param, param_name, data_type=None, exclude_type=None)

    检查参数的数据类型。

    参数：
        - **param** (any) - 待检查的参数。
        - **param_name** (str) - 参数名称。
        - **data_type** (Union[type, tuple[type], list[type], None], 可选) - 允许的数据类型，默认 ``None``。
        - **exclude_type** (Union[type, tuple[type], list[type], None], 可选) - 被排除的数据类型，默认 ``None``。

    异常：
        - **TypeError** - 当 `param` 的数据类型不在允许的数据类型中时抛出。
        - **TypeError** - 当 `param` 的数据类型属于被排除的数据类型时抛出。