mindflow.core.AdaHessian
=========================

.. py:class:: mindflow.core.AdaHessian(*args, **kwargs)

    二阶优化器 AdaHessian，利用 Hessian 矩阵对角元信息进行二阶优化求解。
    有关更多详细信息，请参考论文 `ADAHESSIAN: An Adaptive Second Order Optimizer for Machine Learning <https://arxiv.org/abs/2006.00719>`_ 。
    相关 Torch 版本实现可参考 `Torch 版代码 <https://github.com/amirgholami/adahessian/blob/master/instruction/adahessian.py>`_ 。
    此处 Hessian power 固定为 1，且对 Hessian 对角元做空间平均的方法与 Torch 实现的默认行为一致，描述如下：

    - 对于 1D 张量：不做空间平均；
    - 对于 2D 张量：做行平均；
    - 对于 3D 张量（假设为 1D 卷积）：对最后一个维度做平均；
    - 对于 4D 张量（假设为 2D 卷积）：对最后两个维度做平均。

    参数说明详见 `mindspore.nn.Adam <https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.Adam.html>`_ 。