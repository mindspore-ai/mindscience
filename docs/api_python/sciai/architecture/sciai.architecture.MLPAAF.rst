sciai.architecture.MLPAAF
===========================

.. py:class:: sciai.architecture.MLPAAF(layers, weight_init='xavier_trunc_normal', bias_init='zeros', activation='tanh', last_activation=None, a_value=1.0, scale=1.0, share_type='layer_wise')

    带自适应激活函数的多层感知器。最后一层没有激活函数。

    `layers` 中的第一个值应等于输入Tensor中的最后一个轴的size `in_channels` 。

    有关此改进的 MLP 架构的详细信息，请查看：
    `Locally adaptive activation functions with slope recovery for deep and physics-informed neural networks <https://royalsocietypublishing.org/doi/10.1098/rspa.2020.0334>`_。

    参数：
        - **layers** (Union(tuple[int], list[int])) - 每层神经元数量的列表，例如：[2, 10, 10, 1]。
        - **weight_init** (Union[str, Initializer]) - `Dense` 权重参数的初始化方法。数据类型与 `x` 相同。
          str的值引用自函数 `initializer` 。默认值：'xavier_trunc_normal'。
        - **bias_init** (Union[str, Initializer]) - `Dense` 偏置参数的初始化方法。数据类型与 `x` 相同。
          str的值引用自函数 `initializer` 。默认值：'zeros'。
        - **activation** (Union[str, Cell, Primitive, FunctionType, None]) - 应用于全连接层输出的激活函数，不包括最后一层。可指定激活函数名，
          如 'relu'，或具体激活函数，如 `nn.ReLU()` 。默认值：'tanh'。
        - **last_activation** (Union[str, Cell, Primitive, FunctionType, None]) - 应用于全连接层最后一层输出的激活函数。类型规则与 `activation`
          一致。默认值：None。
        - **a_value** (Union[Number, Tensor, Parameter]) - 自适应可训练参数 `a` 。
        - **scale** (Union[Number, Tensor]) - 固定尺度参数 `scale` 。
        - **share_type** (str) - 自适应函数可训练参数的共享级别，可以是'layer_wise'，'global'。默认值：'layer_wise'。

    输入：
        - **x** (Tensor) - shape为 :math:`(*, in\_channels)` 的Tensor。

    输出：
        Union(Tensor, tuple[Tensor])，网络的输出。

    异常：
        - **TypeError** - `layers` 不是 list、tuple, 或其中任何元素不是整数。
        - **TypeError** - `activation` 不是 str, Cell, Primitive, FunctionType或者None。
        - **TypeError** - `last_activation` 不是 str, Cell, Primitive, FunctionType或者None。
        - **TypeError** - `weight_init` 不是 str、Initializer。
        - **TypeError** - `bias_init` 不是 str、Initializer。
        - **TypeError** - `a_value` 不是 Number, Tensor, Parameter。
        - **TypeError** - `scale` 不是 Number, Tensor。
        - **TypeError** - `share_type` 不是 str。
        - **ValueError** - `share_type` 不支持。

    .. py:method:: sciai.architecture.MLPAAF.a_value()

        获取MLP的局部自适应可训练参数值。

        返回：
            Union(Parameter, tuple[Parameter])，如果 `share_type` 为“global”，则返回全局的可训练参数 `a`；如果为“layer_wise”则返回
            包含所有层的可训练参数 `a` 的列表。