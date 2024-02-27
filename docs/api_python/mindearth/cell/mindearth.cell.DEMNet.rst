mindearth.cell.DEMNet
=========================

.. py:class:: mindearth.cell.DEMNet(in_channels=1, out_channels=256, kernel_size=3, scale=5, num_blocks=42)

    DEM超分辨率模型基于深度残差网络和迁移学习技术。
    有关更多详细信息，请参考论文 `Super-resolution reconstruction of a 3 arc-second global DEM dataset <https://pubmed.ncbi.nlm.nih.gov/36604030/>`_ 。

    参数：
        - **in_channels** (int) - 输入中的通道数。默认值： ``1``。
        - **out_channels** (int) - 输出中的通道数。默认值： ``256``。
        - **kernel_size** (int) - 卷积核尺寸。默认值： ``3``。
        - **scale** (int) - ncoder层的层数。默认值： ``5``。
        - **num_blocks** (int) - 网络中的子模块数目。默认值： ``42``。

    输入：
        - **x** (Tensor) - shape为 :math:`(batch\_size, out\_channels, height\_size, width\_size)` 的Tensor。

    输出：
        Tensor，DEM网络的输出。

        - **output** (Tensor) - shape为 :math:`(batch\_size, out\_channels, new\_height\_size, new_width\_size)` 的Tensor。