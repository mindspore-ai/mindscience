mindchemistry.e3.nn.NormActivation
====================================

.. py:class:: mindchemistry.e3.nn.NormActivation(irreps_in, act, normalize=True, epsilon=None, bias=False, init_method='zeros', dtype=float32, ncon_dtype=float32)

    不可约表示的范数的激活函数。
    对每个不可约表示的范数应用标量激活，并输出该不可约表示的（归一化）版本乘以标量激活的输出。

    参数：
        - **irreps_in** (Union[str, Irrep, Irreps]) - 输入的不可约表示。
        - **act** (Func) - 对 `irreps_in` 的每部分的范数应用的激活函数。
        - **normalize** (bool) - 是否在将输入特征乘以标量之前对其进行归一化。默认值：``True``。
        - **epsilon** (float) - 归一化时，小于 ``epsilon`` 的范数将被钳制到 ``epsilon`` 以避免除零错误。当 `normalize` 为 False 时，不允许设置此参数。默认值：``None``。
        - **bias** (bool) - 是否对 `act` 的输入应用可学习的加性偏置。默认值：``False``。
        - **init_method** (Union[str, float, mindspore.common.initializer]) - 初始化参数的方法。默认值：``'normal'`` 。
        - **dtype** (mindspore.dtype) - 输入张量的类型。默认值：``mindspore.float32`` 。
        - **ncon_dtype** (mindspore.dtype) - ncon 计算模块输入张量的类型。默认值：``mindspore.float32`` 。

    输入：
        - **input** (Tensor) - 形状为 :math:`(..., irreps\_in.dim)` 的张量。

    输出：
        - **output** (Tensor) - 形状为 :math:`(..., irreps\_in.dim)` 的张量。

    异常：
        - **ValueError**: 如果 `epsilon` 不为 None 且 `normalize` 为 False。
        - **ValueError**: 如果 `epsilon` 不是正数。


