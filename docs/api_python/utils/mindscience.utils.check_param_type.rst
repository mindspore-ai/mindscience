mindscience.utils.check_param_type
=================================

.. py:function:: mindscience.utils.check_param_type(param, param_name, data_type=None, exclude_type=None)

    检查参数的数据类型。

    参数：
        - **param** (any) - 待检查的参数。
        - **param_name** (str) - 参数名称（用于错误提示）。
        - **data_type** (Union[type, tuple[type], list[type], None], optional) - 允许的类型；当不为 ``None`` 且 `param` 不是其任一类型实例时抛出异常，默认 ``None``。
        - **exclude_type** (Union[type, tuple[type], list[type], None], optional) - 禁止的类型；当不为 ``None`` 且 `param` 的类型属于其中之一时抛出异常，默认 ``None``。

    异常：
        - **TypeError** - 当 `data_type` 校验失败时抛出（`param` 不是允许类型的实例）。
        - **TypeError** - 当 `exclude_type` 校验失败时抛出（`param` 属于被禁止的类型）。