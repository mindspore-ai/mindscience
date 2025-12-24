mindscience.e3nn.nn.NormActivation
====================================

.. py:class:: mindscience.e3nn.nn.NormActivation(irreps_in, act, normalize=True, epsilon=None, bias=False, init_method='zeros', dtype=mindspore.float32, ncon_dtype=mindspore.float32)

    不可约表示的范数的激活函数。
    对每个不可约表示的范数应用标量激活，并输出该不可约表示的（归一化）版本乘以标量激活的输出。
    可选地，在激活前对范数加上可学习的偏置；并且可将结果按原始范数进行归一化，以保留角度信息，仅调制幅度大小。

    参数：
        - **irreps_in** (Union[str, Irrep, Irreps]) - 输入的不可约表示。
        - **act** (Func) - 对 `irreps_in` 的每部分的范数应用的激活函数。
        - **normalize** (bool，可选) - 是否在将输入特征乘以标量之前对其进行归一化。默认值：``True``。
        - **epsilon** (float，可选) - 归一化时，小于 ``epsilon`` 的范数将被钳制到 ``epsilon`` 以避免除零错误。当 `normalize` 为 False 时，不允许设置此参数。默认值：``None``。
        - **bias** (bool，可选) - 是否对 `act` 的输入应用可学习的加性偏置。默认值：``False``。
        - **init_method** (Union[str, float, mindspore.common.initializer]，可选) - 初始化参数的方法。默认值：``'zeros'`` 。
        - **dtype** (mindspore.dtype，可选) - 输入张量的类型。默认值：``mindspore.float32`` 。
        - **ncon_dtype** (mindspore.dtype，可选) - ncon 计算模块输入张量的类型。默认值：``mindspore.float32`` 。

    输入：
        - **input** (Tensor) - 形状为 :math:`(..., irreps\_in.dim)` 的张量。

    输出：
        - **output** (Tensor) - 形状为 :math:`(..., irreps\_in.dim)` 的张量。

    异常：
        - **ValueError**: 如果 `epsilon` 不为 None 且 `normalize` 为 False。
        - **ValueError**: 如果 `epsilon` 不是正数。


