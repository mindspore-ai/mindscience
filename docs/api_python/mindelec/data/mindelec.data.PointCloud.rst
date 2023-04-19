mindelec.data.PointCloud
========================

.. py:class:: mindelec.data.PointCloud(data_dir, sampling_config, material_config, num_parallel_workers=os.cpu_count())

    读取stp文件以生成PointCloud数据，用于physical-equation模拟。此外，可以分析stp格式的任何三维模型的空间拓扑信息。（CAD中最流行的格式。）

    参数：
        - **data_dir** (str) - stp文件目录，原始数据。
        - **sampling_config** (PointCloudSamplingConfig) - 用于生成PointCloud-Tensor的采样空间配置。
        - **material_config** (MaterialConfig) - 用于生成PointCloud-Tensor的材料解的配置，其中影响材料求解阶段。
        - **num_parallel_workers** (int, 可选) - 并行进程编号，此参数可以对所有计算阶段生效，包括阅读模型、截面构建、空间求解和材料求解。默认值： ``os.cpu_count()``。

    异常：
        - **TypeError** - 如果 `data_dir` 不是str。
        - **TypeError** - 如果 `sampling_config` 不是PointCloudSamplingConfig的实例。
        - **TypeError** - 如果 `material_config` 不是MaterialConfig的实例。
        - **TypeError** - 如果 `num_parallel_workers` 不是int。

    .. py:method:: mindelec.data.PointCloud.model_list()

        获取模型列表。

        返回：
            list，模型列表。

    .. py:method:: mindelec.data.PointCloud.tensor_build()

        使用topology_solving模块中获取的信息构建PointCloud Tensor。如果PointCloud对象使用材料配置初始化，将考虑所有的材料物理信息。所有结果将存储在全局字典列表中，总进程数量num_of_workers用于并行计算。

        返回：
            numpy.ndarray，PointCloud结果。

    .. py:method:: mindelec.data.PointCloud.topology_solving()

        用ray-casting算法求解拓扑空间，对于采样空间中的每个点，我们得到其子模型归属，所有结果都将存储在全局列表中。总进程数量num_of_workers用于并行计算。
