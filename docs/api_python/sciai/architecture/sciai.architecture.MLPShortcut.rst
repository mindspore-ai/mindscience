sciai.architecture.MLPShortcut
============================================

.. py:class:: sciai.architecture.MLPShortcut(layers, weight_init='xavier_trunc_normal', bias_init='zeros', activation='tanh', last_activation=None)

    带有残差的多层感知器。 最后一层没有激活函数。
    更多关于该多层感知器的信息，请参考：
    `Understanding and mitigating gradient pathologies in physics-informed neural networks
    <https://arxiv.org/abs/2001.04536>`_ 。

    参数：
        - **layers** (Union(tuple[int], list[int])) - 每层神经元数量的列表，例如：[2, 10, 10, 1]。
        - **weight_init** (Union(str, Initializer)) - `Dense` 权重参数的初始化方法。数据类型与 `x` 相同。str的值引用自函数 `initializer` 。默认值：'xavier_trunc_normal'。
        - **bias_init** (Union(str, Initializer)) - `Dense` 偏置参数的初始化方法。数据类型与 `x` 相同。str的值引用自函数 `initializer` 。默认值：'zeros'。
        - **activation** (Union(str, Cell, Primitive, FunctionType, None)) - 应用于全连接层输出的激活函数，不包括最后一层。可指定激活函数名，如 'relu'，或具体激活函数，如 `nn.ReLU()` 。默认值：'tanh'。
        - **last_activation** (Union(str, Cell, Primitive, FunctionType, None)) - 应用于全连接层最后一层输出的激活函数。类型规则与 `activation` 一致。默认值：None。

    输入：
        - **x** (Tensor) - 网络的输入Tensor。

    输出：
        Union(Tensor, tuple[Tensor])，网络的输出。

    异常：
        - **TypeError** - `layers` 不是 list、tuple, 或其中任何元素不是整数。
        - **TypeError** - `activation` 不是 str, Cell, Primitive, FunctionType或者None。
        - **TypeError** - `last_activation` 不是 str, Cell, Primitive, FunctionType或者None。
        - **TypeError** - `weight_init` 不是 str、Initializer。
        - **TypeError** - `bias_init` 不是 str、Initializer。