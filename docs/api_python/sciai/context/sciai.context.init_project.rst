sciai.context.init_project
==============================================

.. py:function:: sciai.context.init_project(mode=None, device_id=None, seed=1234, args=None)

    初始化一个项目, 涵盖 `context` 设置、随机种子设置、目录创建和日志级别设置。

    参数：
        - **mode** (Union(int, None)) - ms.PYNATIVE_MODE 用于动态图，ms.GRAPHE_MODE 用于静态图。如果为 ``None`` , 会设置为ms.GRAPH_MODE默认值： ``None`` 。
        - **device_id** (Union(int, None)) - 硬件设备号。默认值： ``None`` 。
        - **seed** (int) - 随机种子。默认值： ``1234`` 。
        - **args** (Union(None, Namespace)) - 参数的命名空间。默认值:  ``None`` 。

    异常：
        - **ValueError** - 如果输入参数不合法。