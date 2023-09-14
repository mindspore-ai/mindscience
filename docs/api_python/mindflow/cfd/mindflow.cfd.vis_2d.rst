mindflow.cfd.vis_2d
=========================

.. py:function:: mindflow.cfd.vis_2d(pri_var, file_name='vis.jpg', permission=stat.S_IREAD + stat.S_IWRITE)

    2d流场可视化。

    参数：
        - **pri_var** (Tensor) - 原始量。
        - **file_name** (str) - 图片文件名，默认值:  ``'vis.jpg'``。
        - **permission** (int) - 文件名权限，默认值:  ``stat.S_IREAD + stat.S_IWRITE``。
