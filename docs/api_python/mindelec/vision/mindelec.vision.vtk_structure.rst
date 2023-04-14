mindelec.vision.vtk_structure
=============================

.. py:function:: mindelec.vision.vtk_structure(grid_tensor, eh_tensor, path_res)

    生成可视化的3D vtk文件。

    参数：
        - **grid_tensor** (numpy.ndarray) - 网格数据，shape为 :math:`(dim_t, dim_x, dim_y, dim_z, 4)`。
        - **eh_tensor** (numpy.ndarray) - 电磁数据，shape为 :math:`(dim_t, dim_x, dim_y, dim_z, 6)`。
        - **path_res** (str) - 输出vtk文件的保存路径。
