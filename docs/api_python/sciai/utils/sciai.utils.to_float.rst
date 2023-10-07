sciai.utils.to_float
=======================

.. py:function:: sciai.utils.to_float(cells, target_type=ms.float32)

    将若干 `Cell` 转换为指定的数据类型。

    参数：
        - **cells** (Union[Cell, list[Cell], tuple[Cell]]) - 要转换的若干个 `Cell`。
        - **target_type** (dtype) - `cells` 将被转换成的目标 MindSpore 数据类型。