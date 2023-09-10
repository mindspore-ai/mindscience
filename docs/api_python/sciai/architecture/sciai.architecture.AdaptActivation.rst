sciai.architecture.AdaptActivation
=====================================

.. py:class:: sciai.architecture.AdaptActivation(activation, a, scale)

    具有可训练参数和固定尺度的自适应激活函数。
    自适应激活函数详情请查看:
    `Adaptive activation functions accelerate convergence in deep and physics-informed neural networks <https://www.sciencedirect.com/science/article/pii/S0021999119308411>`_ 和
    `Locally adaptive activationfunctions with slope recoveryfor deep and physics-informedneural network <https://royalsocietypublishing.org/doi/epdf/10.1098/rspa.2020.0334>`_ 。

    参数：
        - **activation** (Union[str, Cell, Primitive, function]) - 激活函数。
        - **a** (Union[Number, Tensor, Parameter]) - 可训练参数 `a` 。
        - **scale** (Union[Number, Tensor]) - 固定比例参数。

    输入：
        - **x** (Tensor) - AdaptActivation的输入。

    输出：
        Tensor，shape与 `x` 一致的被激活的输出。

    异常：
        - **TypeError** - 如果输入类型不正确。
