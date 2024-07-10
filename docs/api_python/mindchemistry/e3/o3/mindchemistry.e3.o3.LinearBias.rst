mindchemistry.e3.o3.LinearBias
================================

.. py:class:: mindchemistry.e3.o3.LinearBias(irreps_in1, irreps_in2, filter_ir_out, ncon_dtype, **kwargs)

    线性运算等变，可选择添加偏差。
    等效于带有"instructions="linear"的"TensorProduct"，带有添加偏差的选项。关于细节，请参阅"mindchemistry.e3.TensorProduct"。

    参数:
        - **irreps_in1** (Union[str, Irrep, Irreps]) - 第一个输入的Irreps。
        - **irreps_in2** (Union[str, Irrep, Irreps]) - 第二个输入的Irreps。
        - **irrep_norm** (str) - {'component'，'norm'｝，输入和输出表示的假定归一化。默认值:``"component"``。
        - **path_norm** (str) - ｛'element'，'path'｝，路径权重的规范化方法。默认值:``'element'``。
        - **weight_init** (str) - ｛'zeros'，'ones'，'truncatedNormal'，'normal'，'uniform'，'he_uniform'，'she_normal'，'xavier_uniform'}，权重的初始方法。默认值:"normal"。
        - **has_bias** (bool) - 是否将偏差添加到计算中。
        - **ncon_dtype** (mindspore.dtype) - ncon 计算模块输入张量的类型。默认值：``mindspore.float32`` 。
    输入:
        Tensor，LinearBias网络的输入。

        - **v1** (Tensor) - 需要做向量积的第一个向量。
        - **v2** (Tensor) - 需要做向量积的第二个向量。
        - **weight** (Tensor) - 自定义weight。

    输出:
        Tensor，LinearBias网络的输出。

        - **output** (Tensor) - 进行向量积后的结果。