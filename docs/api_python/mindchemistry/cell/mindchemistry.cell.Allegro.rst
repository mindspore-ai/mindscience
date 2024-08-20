mindchemistry.cell.Allegro
============================

.. py:class:: mindchemistry.cell.Allegro(l_max: int = 1, parity_setting="o3_full", num_layers: int = 1, env_embed_multi: int = 8, avg_num_neighbor: float = 1.0, two_body_kwargs=None, latent_kwargs=None, env_embed_kwargs=None, irreps_in=None, enable_mix_precision=False)

    Allegro网络。

    参数：
        - **l_max** (int) - 球谐函数特征的最大阶数。默认值：``1``。
        - **parity_setting** (string) - 对称性相关设置。默认值：``"o3_full"``。
        - **num_layers** (int) - Allegro 网络的层数。默认值：``1``。
        - **env_embed_multi** (int) - 网络中特征的通道数。默认值：``8``。
        - **avg_num_neighbor** (float) - 平均邻近原子数量。默认值：``1.0``。
        - **two_body_kwargs** (dict) - 二体隐层 MLP 的参数。默认值：``None``。
        - **latent_kwargs** (dict) - 隐层 MLP 的参数。默认值：``None``。
        - **env_embed_kwargs** (dict) - 环境嵌入 MLP 的参数。默认值：``None``。
        - **irreps_in** (Irreps) - 输入参数的 irreps 维度。默认值：``None``。
        - **enable_mix_precision** (bool) - 是否启用混合精度。默认值：``False``。

    输入：
        - **embedding_out** (tuple(Tensor)) - 张量元组。
        - **edge_index** (Tensor) - 形状为 :math:`(2, edge\_num)` 的张量。
        - **atom_types** (Tensor) - 张量。

    输出：
        - **output** (Tensor) - 形状为 :math:`(edge\_num, final\_latent\_out)` 的张量。

    异常：
        - **ValueError**: 如果 `irreps_in` 为 None。
        - **ValueError**: 如果 `irreps_in` 中没有必需的字段。
        - **ValueError**: 如果 `input_irreps` 中的乘法错误。
        - **ValueError**: 如果 `env_embed_irreps` 不以标量开头。
        - **ValueError**: 如果 `new_tps_irreps` 的长度与 `tps_irreps` 不等。
        - **ValueError**: 如果 `tps_irreps` 的阶数不为零。
        - **ValueError**: 如果 `full_out_irreps` 的阶数不为零。
        - **ValueError**: 如果 `out_irreps` 的阶数不为零。