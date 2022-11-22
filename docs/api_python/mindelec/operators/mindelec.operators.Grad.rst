mindelec.operators.Grad
=======================

.. py:class:: mindelec.operators.Grad(model, argnum=0)

    计算并返回指定输出列相对于指定输入列的梯度。

    参数：
        - **model** (Cell) - 接受Tensor输入的函数或网络。
        - **argnum** (int) - 指定输出采用的一阶导数的输入。默认值：0。

    输入：
        - **x** (list) - 输入是可变长度参数。第一个输入是二维的网络输入（Tensor），最后三个输入是输入（int）、输出的列索引（int）和网络的输出（Tensor）的列索引。

    输出：
        Tensor。

    异常：
        - **TypeError** - 如果 `argnum` 的类型不是int。
