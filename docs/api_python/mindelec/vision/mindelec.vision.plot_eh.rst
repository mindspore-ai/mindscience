mindelec.vision.plot_eh
=======================

.. py:function:: mindelec.vision.plot_eh(simu_res_tensor, path_image_save, z_index, dpi=300)

    为二维切片绘制每个时间点的电场和磁场值，并将其保存在 `path_image_save` 中。

    参数：
        - **simu_res_tensor** (numpy.ndarray) - 模拟结果。shape为 :math:`(dim_t, dim_x, dim_y, dim_z, 6)`。
        - **path_image_save** (str) - 图像保存路径。
        - **z_index** (int) - 显示z=z_index的2D图像。
        - **dpi** (int) - 图形每英寸的点数。默认值： ``300``。
