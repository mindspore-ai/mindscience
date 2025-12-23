mindscience.models
===================

GraphCast 模型
-----------------

.. mscnautosummary::
    :toctree: models
    :nosignatures:

    mindscience.models.GraphCast.GraphCastNet

注意力模块
-----------------

.. mscnautosummary::
    :toctree: models
    :nosignatures:

    mindscience.models.transformer.Attention
    mindscience.models.transformer.MultiHeadAttention
    mindscience.models.transformer.TransformerBlock

视觉Transformer (ViT)
-------------------------

.. mscnautosummary::
    :toctree: models
    :nosignatures:

    mindscience.models.transformer.VisionTransformer

激活函数
---------------------

.. mscnautosummary::
    :toctree: models
    :nosignatures:

    mindscience.models.layers.get_activation
    mindscience.models.layers.activation.SReLU

基础模块
-------------

.. mscnautosummary::
    :toctree: models
    :nosignatures:

    mindscience.models.layers.DropPath
    mindscience.models.layers.FCSequential
    mindscience.models.layers.InputScale
    mindscience.models.layers.LinearBlock
    mindscience.models.layers.MultiScaleFCSequential
    mindscience.models.layers.ResBlock

UNet2D
------

.. mscnautosummary::
    :toctree: models
    :nosignatures:

    mindscience.models.layers.UNet2D

掩码层
-----------

.. mscnautosummary::
    :toctree: models
    :nosignatures:

    mindscience.models.layers.MaskedLayerNorm

傅里叶神经算子 (FNO)
------------------------------

.. mscnautosummary::
    :toctree: models
    :nosignatures:

    mindscience.models.neural_operator.FNO1D
    mindscience.models.neural_operator.FNO2D
    mindscience.models.neural_operator.FNO3D
    mindscience.models.neural_operator.FNOBlocks

快速傅里叶神经算子 (FFNO)
------------------------------------

.. mscnautosummary::
    :toctree: models
    :nosignatures:

    mindscience.models.neural_operator.FFNO
    mindscience.models.neural_operator.FFNO1D
    mindscience.models.neural_operator.FFNO2D
    mindscience.models.neural_operator.FFNO3D
    mindscience.models.neural_operator.FFNOBlocks
