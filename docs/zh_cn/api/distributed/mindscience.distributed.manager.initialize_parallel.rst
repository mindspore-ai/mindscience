mindscience.distributed.manager.initialize_parallel
===================================================

.. py:function:: initialize_parallel(tensor_parallel_size=1, context_parallel_size=1, order="tp-cp-dp")

    初始化分布式训练的并行通信组。

    此函数创建并初始化分布式训练中不同模型并行（张量、序列、数据和流水线）使用的正交通信组。
    它设置后端通信组，以便代码可以查询每个并行的组大小、排名和名称。调用该函数前需要初始化
    MindSpore通信服务需要的分布式后端。

    参数：
        - **tensor_parallel_size** (int, 可选) - 张量并行的大小。默认值：``1``。
        - **context_parallel_size** (int, 可选) - 序列并行的大小。默认值：``1``。
        - **order** (str, 可选) - 指定计算正交分区时维度顺序的连字符分隔字符串，例如 "tp-cp-dp"。
          顺序决定了所有卡如何分解为用于形成通信组组的多维索引。默认值：``"tp-cp-dp"``。

    异常：
        - **RuntimeError** - 如果卡的总数不能被并行组大小的乘积整除。
