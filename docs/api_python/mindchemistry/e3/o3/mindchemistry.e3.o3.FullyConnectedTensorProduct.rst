mindchemistry.e3.o3.FullyConnectedTensorProduct
======================================================

.. py:class:: mindchemistry.e3.o3.FullyConnectedTensorProduct(irreps_in1, irreps_in2, irreps_out, ncon_dtype, **kwargs)

    全连接加权张量积。所有满足 :math:`|l_1 - l_2| \leq l_{out} \leq l_1 + l_2` 的路径都将被考虑。
    等效于 `TensorProduct` 的 `instructions='connect'`。详细信息请参见 `mindchemistry.e3.TensorProduct`。

    参数：
        - **irreps_in1** (Union[str, Irrep, Irreps]): 第一个输入的 Irreps。
        - **irreps_in2** (Union[str, Irrep, Irreps]): 第二个输入的 Irreps。
        - **irreps_out** (Union[str, Irrep, Irreps]): 输出的 Irreps。
        - **irrep_norm** (str): {'component', 'norm'}，输入和输出表示的假定归一化。默认值：``component``。
        - **path_norm** (str): {'element', 'path'}，路径权重的归一化方法。默认值：``element``。
        - **weight_init** (str): {'zeros', 'ones', 'truncatedNormal', 'normal', 'uniform', 'he_uniform', 'he_normal', 'xavier_uniform'}，权重的初始化方法。默认值：``normal``。
        - **ncon_dtype** (mindspore.dtype): ncon 计算模块输入张量的类型。默认值：``mindspore.float32``。
