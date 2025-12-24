mindscience.e3nn.o3.compose_angles
=========================================

.. py:function:: mindscience.e3nn.o3.compose_angles(a1, b1, c1, a2, b2, c2)

    计算由两个旋转组合而成的欧拉角。

    给定由欧拉角 (a1, b1, c1) 和 (a2, b2, c2) 表示的两个旋转，
    此函数返回组合旋转的欧拉角 (a, b, c)。

    .. math::

        R(a, b, c) = R(a_1, b_1, c_1) \circ R(a_2, b_2, c_2)

    .. note::
        第二组欧拉角 `a2, b2, c2` 首先应用，而第一组欧拉角 `a1, b1, c1` 随后应用。
        欧拉角的元素应为以下类型之一：float, float32, np.float32。

    参数：
        - **a1** (Union[Tensor[float32], list[float], tuple[float], ndarray[np.float32], float]) - 第二次应用的 alpha 欧拉角。
        - **b1** (Union[Tensor[float32], list[float], tuple[float], ndarray[np.float32], float]) - 第二次应用的 beta 欧拉角。
        - **c1** (Union[Tensor[float32], list[float], tuple[float], ndarray[np.float32], float]) - 第二次应用的 gamma 欧拉角。
        - **a2** (Union[Tensor[float32], list[float], tuple[float], ndarray[np.float32], float]) - 第一次应用的 alpha 欧拉角。
        - **b2** (Union[Tensor[float32], list[float], tuple[float], ndarray[np.float32], float]) - 第一次应用的 beta 欧拉角。
        - **c2** (Union[Tensor[float32], list[float], tuple[float], ndarray[np.float32], float]) - 第一次应用的 gamma 欧拉角。

    返回：
        tuple[Tensor]，由组合后的 :math:`\alpha` 、 :math:`\beta` 、 :math:`\gamma` 组成的三元组。
