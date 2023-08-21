mindsponge.cell.InvariantPointAttention
=======================================

.. py:class:: mindsponge.cell.InvariantPointAttention(num_head, num_scalar_qk, num_scalar_v, num_point_v, num_point_qk, num_channel, pair_dim)

    该模块用于更新序列表示（即输入inputs_1d），在序列表示中加入位置信息。
    其中注意力由三部分构成，即由序列表示得到的q, k, v，由序列表示与刚体变换组局部坐标系T交互得到的q', k', v'，
    以及从氨基酸对表示（输入中的inputs_2d）中得到的偏移b。

    .. math::
        a_{ij} = Softmax(w_l(c_1{q_i}^Tk_j+b{ij}-c_2\sum {\left \| T_i\circ q'_i-T_j\circ k'_j \right \| ^{2 } })

    其中i,j分别表示序列中第i、第j个氨基酸，T即输入中的rotation和translation。

    参考文献：`Jumper et al. (2021) Suppl. Alg. 22 InvariantPointAttention <https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-021-03819-2/MediaObjects/41586_2021_3819_MOESM1_ESM.pdf>`_。

    参数：
        - **num_head** (int) - 头的数量。
        - **num_scalar_qk** (int) - scalar query/key的数量。
        - **num_scalar_v** (int) - scalar value的数量。
        - **num_point_v** (int) - point value的数量。
        - **num_point_qk** (int) - point query/key的数量。
        - **num_channel** (int) - 通道数量。
        - **pair_dim** (int) - pair的最后一维长度。

    输入：
        - **inputs_1d** (Tensor) - Evoformer模块的输出msa表示矩阵中的第一行，也即序列表示, :math:`[N_{res}, num\_channel]` 。
        - **inputs_2d** (Tensor) - Evoformor模块的输出氨基酸对表示矩阵, :math:`[N_{res}, N_{res}, pair\_dim]` 。
        - **mask** (Tensor) - 掩码，表示inputs_1d的哪些元素参与了attention, :math:`[N_{res}, 1]` 。
        - **rotation** (tuple) - 刚体变换组局部坐标系 :math:`T(r,t)` 中的旋转信息, 长度为9的元组，每个元素shape为 :math:`[N_{res}]` 。
        - **translation** (tuple) - 刚体变换组局部坐标系 :math:`T(r,t)` 中的旋转信息的偏移信息, 长度为3的元组，每个元素shape为 :math:`[N_{res}]` 。

    输出：
        Tensor，input_1d的更新，shape为 :math:`[N_{res}, num\_channel]` 。