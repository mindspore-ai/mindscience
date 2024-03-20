mindflow.cell.PeRCNN
=========================

.. py:class:: mindflow.cell.PeRCNN(dim, in_channels, hidden_channels, kernel_size, dt, nu, laplace_kernel=None, conv_layers_num=3, padding="periodic", compute_dtype=ms.float32)

    物理编码循环卷积神经网络(PeRCNN)，对给定的物理结构进行强制编码，实现稀疏数据上的时空演化的学习。PeRCNN可以应用于关于PDE系统的各种问题，包括正向和反向分析、数据驱动建模和PDE的发现。
    更多信息可参考论文 `Encoding physics to learn reaction–diffusion processes <https://www.nature.com/articles/s42256-023-00685-7>`_ 。
    在本网络中，lazy_inline用于编译阶段的加速，但当前其仅在昇腾后端生效。
    PeRCNN目前支持带两个物理结构的输入。当输入形状不同时，用户需要手动增加或移除相关的参数和pi_blocks。

    参数：
        - **dim** (int) - 输入的物理维度，二维输入的shape长度为4，三维为5，数据遵循 `\"NCHW\"` 或 `\"NCDHW\"` 格式。
        - **in_channels** (int) - 输入空间的通道数。
        - **hidden_channels** (int) - 并行卷积层中的输出空间的通道数。
        - **kernel_size** (int) - 并行卷积层中的卷积核参数。
        - **dt** (Union[int, float]) - PeRCNN的时间步。
        - **nu** (Union[int, float]) - 扩散项的系数。
        - **laplace_kernel** (mindspore.Tensor) - 三维下，设置核的shape为 :math:`(\text{kernel_size[0]}, \text{kernel_size[1]}, \text{kernel_size[2]})` ，则shape为 :math:`(C_{out}, C_{in}, \text{kernel_size[0]}, \text{kernel_size[1]}, \text{kernel_size[1]})` 。二维下，shape向量为 :math:`(N, C_{in} / \text{groups}, \text{kernel_size[0]}, \text{kernel_size[1]})` ，则核的shape为 :math:`(\text{kernel_size[0]}, \text{kernel_size[1]})` 。
        - **conv_layers_num** (int) - 并行卷积层的数量。默认值： ``3``。
        - **padding** (str) - 边界填充，当前仅支持周期填充。默认值： ``periodic``。
        - **compute_dtype** (dtype.Number) - PeRCNN的数据类型。默认值： ``mindspore.float32`` 。支持以下数据类型： ``mindspore.float32`` 或 ``mindspore.float16``。GPU后端建议使用float32，Ascend后端建议使用float16。

    输入：
        - **x** (Tensor) - 三维下的shape为 :math:`(batch\_size, channels, depth, height, width)` ，二维下的shape为 :math:`(batch\_size, channels, height, width)` 。

    输出：
        Tensor，与输入的shape一致。

    异常：
        - **TypeError** - 如果 `dim` 、 `in_channels` 、 `hidden_channels` 、 `kernel_size` 不是int。
        - **TypeError** - 如果 `dt` 和 `nu` 既不是int也不是float。
