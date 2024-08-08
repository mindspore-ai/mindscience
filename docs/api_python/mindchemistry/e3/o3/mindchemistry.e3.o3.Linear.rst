mindchemistry.e3.o3.Linear
============================

.. py:class:: mindchemistry.e3.o3.Linear(irreps_in, irreps_out, ncon_dtype=float32, **kwargs)

    线性等变操作。
    等效于"instructions='linear'"的"TensorProduct'"。有关详细信息，请参阅 :class:`mindchemistry.e3.o3.TensorProduct`。

    参数：
        - **irreps_in** (Union[str, Irrep, Irreps]) - 输入的Irreps。
        - **irreps_out** (Union[str, Irrep, Irreps]) - 输出的Irreps。
        - **irrep_norm** (str) - {'component'，'norm'｝，输入和输出表示的假定归一化。默认值: ``"component"``。
        - **path_norm** (str) - ｛'element'，'path'｝，路径权重的规范化方法。默认值:``'element'``。
        - **weight_init** (str) - ｛'zeros'，'ones'，'truncatedNormal'，'normal'，'uniform'，'he_uniform'，'she_normal'，'xavier_uniform'}，权重的初始方法。默认值:``"normal"``。
        - **ncon_dtype** (mindspore.dtype) - ncon 计算模块输入张量的类型。默认值：``mindspore.float32`` 。