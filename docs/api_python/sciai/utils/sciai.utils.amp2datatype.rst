sciai.utils.amp2datatype
==============================================

.. py:function:: sciai.utils.amp2datatype(type_str)

    从自动混合精度字符串到 `MindSpore` 数据类型的映射。支持输入为 `O0` 至 `O3` 自动混合精度等级。

    参数：
        - **type_str** (str) - 自动混合精度字符串。

    返回：
        - **dtype** - `MindSpore` 数据类型。