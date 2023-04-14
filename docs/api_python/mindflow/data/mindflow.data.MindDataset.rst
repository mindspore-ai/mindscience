mindflow.data.MindDataset
=========================

.. py:class:: mindflow.data.MindDataset(dataset_files, dataset_name="dataset", constraint_type="Label", shuffle=True, num_shards=None, shard_id=None, sampler=None, num_samples=None, num_parallel_workers=None)

    从MindRecord类型的数据创建数据集。

    参数：
        - **dataset_files** (Union[str, list[str]]) - 如果数据集文件是str，则它代表MindRecord的一个子文件名。同一路径下来自同一个数据源的子文件会被自动加载。如果 `dataset_file` 是list，它表示要读取的数据集文件列表。
        - **dataset_name** (str, 可选) - 数据集名称，默认值： ``"dataset_name"``。
        - **constraint_type** (str, 可选) - 指定数据集的约束类型，以获取其相应的损失函数。默认值： ``"Label"``。其他支持的类型可详见 `mindflow.data.Dataset`。
        - **shuffle** (Union[bool, Shuffle level], 可选) - 每个epoch对数据执行shuffle。如果shuffle为 ``False``，则不执行shuffle。如果shuffle为 ``True``，则执行全局shuffle。默认值： ``True``。
          而且，有两种shuffle level：
        
          - ``Shuffle.GLOBAL``：对文件和样例进行shuffle。
          - ``Shuffle.FILES``：仅对文件shuffle。

        - **num_shards** (int, 可选) - 数据集将划分为的分片数指定此参数时， `num_samples` 反映每个分片的最大样本数。默认值： ``None``。
        - **shard_id** (int, 可选) -  `num_shards` 中的分片ID。只有当同时指定 `num_shards` 时，才能指定该参数。默认值： ``None``。
        - **sampler** (Sampler, 可选) - 用于从数据集。支持列表：SubsetRandomSampler、PkSampler、RandomSampler、SequentialSampler、DistributedSampler。默认值： ``None``，采样器是独占的使用shuffle和block_reader。
        - **num_samples** (int, 可选) - 要包括在数据集中的样本数。默认值： ``None``，所有样本。
        - **num_parallel_workers** (int, 可选) - 读取器的数。默认值： ``None``。

    异常：
        - **ValueError** - 如果 `dataset_files` 无效或不存在。
        - **TypeError** - 如果数据集名称不是string。
        - **ValueError** - 如果constraint_type.lower()不在[``“equation”``, ``“bc”``, ``“ic”``, ``“label”``, ``“function”``, ``“custom”``]中。
        - **RuntimeError** - 如果指定了 `num_shards` ，但 `shard_id` 为 ``None``。
        - **RuntimeError** - 如果指定了 `shard_id` ，但 `num_shards` 为 ``None``。
        - **ValueError** - 如果 `shard_id` 无效（<0或>= `num_shards`）。

    .. py:method:: create_dataset(batch_size=1, preprocess_fn=None, updated_columns_list=None, drop_remainder=True, prebatched_data=False, num_parallel_workers=1, python_multiprocessing=False)

        创建最终的MindSpore类型数据集。

        参数：
            - **batch_size** (int, 可选) - 每个批处理创建的行数为int。默认值： ``1``。
            - **preprocess_fn** (Union[list[TensorOp], list[functions]], 可选) - 要进行的操作列表应用于数据集。操作按它们在此列表中的显示顺序应用。默认值： ``None``。
            - **updated_columns_list** (list, 可选) - 对数据集的列进行的操作。默认值： ``None``。
            - **drop_remainder** (bool, 可选) - 确定是否删除最后一个块，其数据行号小于批处理大小。如果为 ``True``，如果有更少的比批处理大小行可用于创建最后一个批处理，那么这些行将被丢弃，而不传播到子节点。默认值： ``True``。
            - **prebatched_data** (bool, 可选) - 在数据预处理前生成预批处理数据。默认值： ``False``。
            - **num_parallel_workers** (int, 可选) - 并行处理数据集的工作线程（线程）数量。默认值： ``1``。
            - **python_multiprocessing** (bool, 可选) - 使用多处理并行Python函数per_batch_map。如果函数计算量很大，此选项可能会很有用。默认值： ``False``。

        返回：
            BatchDataset，批处理的数据集。

    .. py:method:: get_columns_list()

        获取数据集中的列。

        返回：
            list[str]。最终统一数据集的列名列表。

    .. py:method:: set_constraint_type(constraint_type='Equation')

        设置数据集的约束类型。

        参数：
            - **constraint_type** (Union[str, dict]) - 指定数据集的约束类型。如果是字符串，则约束所有子数据集的类型将设置为相同的类型。如果是dict，则子数据集及其约束类型由对(key, value)指定。默认值： ``“Equation”``。

    .. py:method:: split_dataset(dataset_dict, constraint_dict=None)

        拆分原始数据集以设置差异损失函数。

        参数：
            - **dataset_dict** (dict) - 每个子数据集的字典，key是标记的名称，而value 指子数据集中包含的指定列。
            - **constraint_dict** (Union[None, str, dict]) - 指定数据集的约束类型。如果是 ``None``，则 ``“Label”`` 将为所有人设置。如果是字符串，则所有将设置为相同的字符串。如果是dict，子数据集及其约束类型由对(key, value)指定。默认值： ``None``。