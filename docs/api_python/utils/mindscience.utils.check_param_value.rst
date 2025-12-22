mindscience.utils.check_param_value
==================================

.. py:function:: mindscience.utils.check_param_value(param, param_name, valid_value)

    检查参数取值是否在允许范围内。

    参数：
        - **param** (any) - 待检查的参数值。
        - **param_name** (str) - 参数名称（用于错误提示）。
        - **valid_value** (Union[any, tuple, list]) - 允许的取值集合。

    异常：
        - **ValueError** - 当 `param` 的取值不在允许的取值中时抛出。