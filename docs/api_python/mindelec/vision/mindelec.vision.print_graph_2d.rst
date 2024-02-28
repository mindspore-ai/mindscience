mindelec.vision.print_graph_2d
==============================

.. py:function:: mindelec.vision.print_graph_2d(name, x, y, path, clear=True)

    绘制二维散点图。

    参数：
        - **name** (str) - 图形的名称。
        - **x** (numpy.ndarray) - 要绘制的数据x，shape为 :math:`(dim\_print,)`。
        - **y** (numpy.ndarray) - 要绘制的数据y，shape为 :math:`(dim\_print,)`。
        - **path** (str) - 图形的保存路径。
        - **clear** (bool) - 指定是否清除当前轴。默认值： ``True``。
