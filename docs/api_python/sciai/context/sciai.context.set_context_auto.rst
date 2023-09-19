sciai.context.set_context_auto
======================================================

.. py:function:: sciai.context.set_context_auto(mode=ms.GRAPH_MODE, device_id=None)

    将 `ms.context` 设置为指定模式，自动识别平台。 如果 `card` 是None，则不会设置设备号。

    参数：
        - **mode** (int) - Mindspore运行模式，可以是ms.PYNATIVE_MODE或ms.GRAPH_MODE。 默认值：ms.GRAPH_MODE。
        - **device_id** (Union(int, None)) - 如果指定，则设置运行设备号。默认值：None。

    异常：
        - **ValueError** - 如果 `device_id` 是非法值。