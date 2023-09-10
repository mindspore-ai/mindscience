sciai.architecture.PDENet
=========================

.. py:class:: sciai.architecture.PDENet(height, width, channels, kernel_size, max_order, dx=0.01, dy=0.01, dt=0.01, periodic=True, enable_moment=True, if_fronzen=False)

    PDE-Net模型。
    PDE-Net是一个前馈深度网络，可同时实现两个目标：准确预测复杂的系统，并揭示底层隐藏的PDE模型。基本思想是学习微分算子通过学习卷积核（过滤器），并将神经网络或其他机器学习方法应用于
    近似未知非线性响应。PDE-Net的特殊性在于，卷积核受“矩”的约束，这使得模型能够轻松地识别PDE模型，同时仍保持网络的表达能力和预测能力。
    这些约束通过充分利用微分算子的阶数与卷积核的关系得到的.一个重要的概念起源于小波理论。有关更多详细信息，请参考论文 `PDE-NET: LEARNING PDES FROM DATA <https://arxiv.org/pdf/1710.09668.pdf>`_ 。

    参数：
        - **height** (int) - PDE-Net输入和输出Tensor的高度。
        - **width** (int) - PDE-Net输入和输出Tensor的宽度。
        - **channels** (int) - PDE-Net输入和输出Tensor的通。
        - **kernel_size** (int) - 指定2D卷积内核的高度和宽度。
        - **max_order** (int) - PDE模型的最大顺序。
        - **dx** (float) - x维的空间分辨率。默认值： ``0.01``。
        - **dy** (float) - y维的空间分辨率。默认值： ``0.01``。
        - **dt** (float) - PDE-Net的时间步长。默认值： ``0.01``。
        - **periodic** (bool) - 指定周期是否与卷积核一起使用。默认值： ``True``。
        - **enable_moment** (bool) - 指定卷积核是否受moment约束。默认值： ``True``。
        - **if_fronzen** (bool) - moment里的参数是否参与训练。默认值： ``False``。

    输入：
        - **input** (Tensor) - shape为 :math:`(batch\_size, channels, height, width)` 的Tensor。

    输出：
        Tensor，具有与 `input` 相同的shape，数据类型为float32。

    异常：
        - **TypeError** - 如果 `height` 、 `width` 、 `channels` 、 `kernel_size` 或 `max_order` 不是int。
        - **TypeError** - 如果 `periodic` 、 `enable_moment` 、 `if_fronzen` 不是bool。
