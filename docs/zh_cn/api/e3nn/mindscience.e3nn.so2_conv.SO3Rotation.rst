mindscience.e3nn.so2_conv.SO3Rotation
=====================================

.. py:class:: mindscience.e3nn.so2_conv.SO3Rotation(lmax, irreps_in, irreps_out)

    处理球面谐波不可约表示的 SO(3) 旋转的类。

    参数：
        - **lmax** (int) - 考虑的最大角动量阶数。
        - **irreps_in** (Union[str, Irreps]) - 输入不可约表示。
        - **irreps_out** (Union[str, Irreps]) - 输出不可约表示。

    .. py:method:: narrow(inputs, axis, start, length)

        沿指定维度对张量进行切片。

        参数：
            - **inputs** (Tensor) - 待切片的张量。
            - **axis** (int) - 进行切片的维度。
            - **start** (int) - 切片起始位置索引。
            - **length** (int) - 切片包含的元素数量。

        返回：
            Tensor，切片后的张量。
    
    .. py:method:: rotate(embedding, wigner)

        按提供的 Wigner-D 矩阵对嵌入张量进行 SO(3) 旋转。

        参数：
            - **embedding** (Tensor) - 待旋转的输入张量，形状为 ``(..., irreps_in.dim)``，包含待旋转的球谐系数。
            - **wigner** (tuple[Tensor]) - `l = 0 … lmax` 的 Wigner-D 矩阵元组，每个形状为 ``(..., 2l+1, 2l+1)``。

        返回：
            tuple[Tensor]，与 `irreps_in` 中每个 irrep 对应的一组旋转后张量，每个形状为 ``(..., mul, 2l+1)``。

    .. py:method:: rotate_inv(embedding, wigner_inv)

        使用提供的逆 Wigner-D 矩阵对嵌入张量进行逆旋转。

        参数：
            - **embedding** (tuple[Tensor]) - 与 `irreps_out` 中每个 irrep 对应的一组张量，每个形状为 ``(..., mul, 2l+1)``。
            - **wigner_inv** (tuple[Tensor]) - `l = 0 … lmax` 的逆（转置）Wigner-D 矩阵元组，每个形状为 ``(..., 2l+1, 2l+1)``。

        返回：
            Tensor，逆旋转并拼接后的输出张量，形状为 ``(..., irreps_out.dim)``。

    .. py:method:: rotation_to_wigner_d_matrix(edge_rot_mat, start_lmax, end_lmax)

        将批量 :math:`3 \times 3` 旋转矩阵转换为指定角动量范围的 Wigner-D 矩阵。

        参数：
            - **edge_rot_mat** (Tensor) - 旋转矩阵批次，形状为 ``(..., 3, 3)``。
            - **start_lmax** (int) - 最小角动量阶数。
            - **end_lmax** (int) - 最大角动量阶数。

        返回：
            list[Tensor]，对应从 `start_lmax` 到 `end_lmax` 的各阶 Wigner-D 矩阵，每个张量形状为 ``(..., 2l+1, 2l+1)``。
        
    .. py:method:: set_wigner(rot_mat3x3)

        根据批量 :math:`3 \times 3` 旋转矩阵计算 Wigner-D 矩阵及其逆（转置）。

        参数：
            - **rot_mat3x3** (Tensor) - 旋转矩阵批次，形状为 ``(..., 3, 3)``。

        返回：
            tuple[list[Tensor], list[Tensor]]，包含两组列表：
            
            - wigner：`l = 0 … lmax` 的 Wigner-D 矩阵，每个形状为 ``(..., 2l+1, 2l+1)``。
            - wigner_inv：对应的转置（逆）矩阵，形状同上。