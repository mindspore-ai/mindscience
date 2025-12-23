mindscience.e3nn.so2_conv.init_edge_rot_mat
===========================================

.. py:function:: mindscience.e3nn.so2_conv.init_edge_rot_mat(edge_distance_vec)

    根据边距向量初始化旋转矩阵。

    参数：
        - **edge_distance_vec** (Tensor) - 边距离向量，形状为 ``(batch_size, 3)``。

    返回：
        Tensor，旋转矩阵，形状为 ``(batch_size, 3, 3)``。