mindchemistry.e3.o3.compose_angles
=========================================

.. py:function:: mindchemistry.e3.o3.compose_angles(a1, b1, c1, a2, b2, c2)

    计算两个欧拉角集合的组合欧拉角。

    .. math::

        R(a, b, c) = R(a_1, b_1, c_1) \circ R(a_2, b_2, c_2)

    注意：
        - 第二组欧拉角 `a2, b2, c2` 首先应用，而第一组欧拉角 `a1, b1, c1` 随后应用。
        - 欧拉角的元素应为以下类型之一：float, float32, np.float32。

    参数：
        - **a1** (Union[Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]): 第二次应用的 alpha 欧拉角。
        - **b1** (Union[Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]): 第二次应用的 beta 欧拉角。
        - **c1** (Union[Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]): 第二次应用的 gamma 欧拉角。
        - **a2** (Union[Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]): 第一次应用的 alpha 欧拉角。
        - **b2** (Union[Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]): 第一次应用的 beta 欧拉角。
        - **c2** (Union[Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]): 第一次应用的 gamma 欧拉角。

    返回：
        - **alpha** (Tensor) - 组合后的 alpha 欧拉角。
        - **beta** (Tensor) - 组合后的 beta 欧拉角。
        - **gamma** (Tensor) - 组合后的 gamma 欧拉角。
