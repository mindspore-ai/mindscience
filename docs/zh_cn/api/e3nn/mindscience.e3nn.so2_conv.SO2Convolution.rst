mindscience.e3nn.so2_conv.SO2Convolution
========================================

.. py:class:: mindscience.e3nn.so2_conv.SO2Convolution(irreps_in, irreps_out)

    面向圆群的复值特征的 SO(2) 等变卷积层。

    该层在两个 `Irreps` 空间之间映射，描述输入/输出在平面旋转下的变换方式。保持 m 量子数（SO(2) 不可约表示的索引）对角不混合，使每个 m 块独立处理。对于 :math:`m = 0` （标量部分）使用实值全连接层；对于 :math:`m > 0` 使用复值 :math:`1\times 1` 卷积（以作用在实/虚部上的实值 :math:`2\times 2` 权重矩阵实现）。

    参数：
        - **irreps_in** (Union[str, Irreps]) - 输入不可约表示，例如 ``"3x0e + 2x1o"``（3 个标量 + 2 个向量）。
        - **irreps_out** (Union[str, Irreps]) - 输出不可约表示，例如 ``"5x0e + 1x1o"``。

    输入：
        - **x** (list[Tensor]) - 实值张量列表，每个张量形状为 ``(batch, mul, 2l+1)``，表示每个阶数 ``l`` 的复值 SO(2) 特征；最后一维为磁量子数索引 ``m = -l ... +l``。
        - **x_edge** (Tensor) - 形状为 ``(edges, features)`` 的实值张量，包含在卷积过程中广播并与 SO(2) 特征结合的边（径向）属性。

    输出：
        - **tuple** (Tensor) - 与 ``irreps_out`` 中每个 irrep 对应的一组实值张量；每个张量形状为 ``(batch, mul, 2l+1)``，包含对应阶数 ``l`` 的复值 SO(2) 特征；最后一维为磁量子数索引 ``m = -l ... +l``。