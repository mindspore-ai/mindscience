mindscience.common.get_2d_sin_cos_pos_embed
============================================

.. py:function:: mindscience.common.get_2d_sin_cos_pos_embed(embed_dim, grid_size)

    在二维网格上构造二维正弦-余弦位置编码。

    参数：
        - **embed_dim** (int) - 每个位置的输出维度。
        - **grid_size** (tuple(int)) - 网格的高度和宽度。

    返回：
        Numpy.array，形状为 :math:`(1, grid\_height*grid\_width, embed\_dim)` 的数组。