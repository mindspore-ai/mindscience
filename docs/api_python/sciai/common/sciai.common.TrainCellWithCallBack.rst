sciai.common.TrainCellWithCallBack
============================================

.. py:class:: sciai.common.TrainCellWithCallBack(network, optimizer, loss_interval=1, time_interval=0, ckpt_interval=0, loss_names=("loss",), batch_num=1, grad_first=False, amp_level="O0", ckpt_dir="./checkpoints", clip_grad=False, clip_norm=1e-3, model_name="model")

    带有回调的 TrainOneStepCell，可以处理多重损失。 回调功能如下：

    1.loss：打印损失。
    2.time：打印步骤所花费的时间，以及从开始所花费的时间。
    3.ckpt：保存checkpoint。

    参数：
        - **network** (Cell) - 训练网络。该网络支持多个输出。
        - **optimizer** (Cell) - 用于更新网络参数的优化器。
        - **loss_interval** (int) - 打印loss的步长间隔。 如果为 0，则不会打印loss。 默认值：1。
        - **time_interval** (int) - 打印时间的步长间隔。 如果为 0，则不会打印时间。 默认值：0。
        - **ckpt_interval** (int) - 保存checkpoint的epoch间隔，根据 `batch_num` 计算，如果为0，则不会保存checkpoint。 默认值：0。
          如果是n个，则每个字符串对应同一位置的loss；如果是n+1个，第一个损失名称代表所有输出的总和，其他一一对应。默认值：("loss",)。
        - **loss_names** (Union(str, tuple[str], list[str])) - 各损失的名字，按照网络输出的顺序排列。 它可以接受n个或n+1个字符串，
          其中n为网络输出的个数。如果n个，每个字符串对应同一位置的loss；如果n+1个，第一个字符串为所有输出的总和的损失名。 默认值：(“loss”,)。
        - **batch_num** (int) - 每个时期有多少批次。 默认值：1。
        - **grad_first** (bool) - 若为True，则只有网络的第一个输出参与梯度下降。 否则所有输出之和参与梯度下降。默认值：False。
        - **amp_level** (str) - 混合精度等级，目前支持["O0", "O1", "O2", "O3"]. 默认值："O0".
        - **ckpt_dir** (str) - checkpoint保存路径。 默认值："./checkpoints"。
        - **clip_grad** (bool) - 是否裁剪梯度。默认值：False.
        - **clip_norm** (Union(float, int)) - 梯度裁剪率，需为正数. 仅当 `clip_grad` 为True时生效. 默认值：1e-3.
        - **model_name** (str) - 模型名，影响ckpt名字。 默认："model"。

    输入：
        - **\*args** (tuple[Tensor]) - 网络输入张量的元组.

    输出：
        Union(Tensor, tuple[Tensor]) - 网络输出的单项或多项loss.

    异常：
        - **TypeError** - 如果输入参数不是要求的类型。

    .. py:method:: sciai.common.TrainCellWithCallBack.calc_ckpt_name(iter_str, model_name, postfix="")
        :staticmethod:

        计算检查点文件名。

        参数：
            - **iter_str** (Union[str]) - 迭代次数或epoch数。
            - **model_name** (str) - 模型名称。
            - **postfix** (str) - 文件名后缀。 默认：""。

        返回：
            str，checkpoint的文件名。

    .. py:method:: sciai.common.TrainCellWithCallBack.calc_optim_ckpt_name(model_name, postfix="")
        :staticmethod:

        计算最新的检查点文件名。

        参数：
            - **model_name** (str) - 模型名称。
            - **postfix** (str) - 文件名后缀。 默认：""。

        返回：
            str，checkpoint的文件名。