sciai.common.lbfgs_train
============================================

.. py:function:: sciai.common.lbfgs_train(loss_net, input_data, lbfgs_iter)

    L-BFGS训练函数，目前只能在PYNATIVE模式下运行。

    参数：
        - **loss_net** (Cell) - 返回loss作为目标函数的网络。
        - **input_data** (Union[Tensor, tuple[Tensor]]) - loss_net的输入数据。
        - **lbfgs_iter** (int) - l-bfgs训练过程的迭代次数。