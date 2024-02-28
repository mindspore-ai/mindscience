mindelec.vision.plot_s11
========================

.. py:function:: mindelec.vision.plot_s11(s11_tensor, path_image_save, legend, dpi=300)

    绘制s11频率曲线并将其保存在 `path_image_save` 中。

    参数：
        - **s11_tensor** (numpy.ndarray) - s11数据，shape为 :math:`(dim\_frequency, 2)`。
        - **path_image_save** (str) - s11频率曲线保存路径。
        - **legend** (str) - s11的图例，绘制参数。
        - **dpi** (int) - 图形每英寸的点数。默认值： ``300``。
