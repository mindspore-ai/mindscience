sciai.utils.make_sciai_dirs
==============================================

.. py:function:: sciai.utils.make_sciai_dirs()

    为 sciai 项目创建目录。 如果 `checkpoints` 、 `data` 、 `figures` 、 `logs` 任意目录不存在，则会创建对应目录。
    `checkpoints` 保存模型检查点； `data` 存放数据集与生成数据； `figures` 保存作图结果； `logs` 存放训练与验证过程的日志。