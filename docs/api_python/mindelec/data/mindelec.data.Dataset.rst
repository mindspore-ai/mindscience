mindelec.data.Dataset
=====================

.. py:class:: mindelec.data.Dataset(geometry_dict=None, existed_data_list=None, dataset_list=None)

    将数据集合并在一起。

    参数：
        - **geometry_dict** (dict, 可选) - 指定要合并的几何数据集。键为几何实例，值为几何体类型的列表。例如，geometry_dict = {geom : [``"domain"``, ``"BC"``, ``"IC"``]}。默认值： ``None``。
        - **existed_data_list** (Union[list, tuple, ExistedDataConfig], 可选) - 指定要合并的现有数据集。例如，existed_data_list = [``ExistedDataConfig_Instance1``, ``ExistedDataConfig_Instance2``]。默认值： ``None``。
        - **dataset_list** (Union[list, tuple, Data], 可选) - 指定要合并的数据实例。例如，dataset_list=[``BoundaryIC_Instance``, ``Equation_Instance``, ``BoundaryBC_Instance``, ``ExistedData_Instance``]。默认值： ``None``。

    异常：
        - **ValueError** - 如果 `geometry_dict` 、 `existed_data_list` 和 `dataset_list` 都为 ``None``。
        - **TypeError** - 如果 `geometry_dict` 的类型不是dict。
        - **TypeError** - 如果 `geometry_dict` 的键类型不是mindelec.geometry.Geometry的实例。
        - **TypeError** - 如果 `existed_data_list` 的类型不是列表、元组或ExistedDataConfig的实例。
        - **TypeError** - 如果 `existed_data_list` 的元素不是ExistedDataConfig的实例。
        - **TypeError** - 如果 `dataset_list` 的元素不是Data的实例。

    .. py:method:: mindelec.data.Dataset.create_dataset(batch_size=1, preprocess_fn=None, input_output_columns_map=None, shuffle=True, drop_remainder=True, prebatched_data=False, num_parallel_workers=1, num_shards=None, shard_id=None, python_multiprocessing=False)

        创建最终的MindSpore类型数据集以合并所有子数据集。

        参数：
            - **batch_size** (int, 可选) - 每个批处理创建的行数，int值。默认值： ``1``。
            - **preprocess_fn** (Union[list[TensorOp], list[functions]], 可选) - 要应用于数据集的进行操作的列表。按它们在此列表中的顺序遍历操作。默认值： ``None``。
            - **input_output_columns_map** (dict, 可选) - 指定要替换的列和需要替换成的内容。键是要被替换的列名，值是要替换成的内容。如果映射后所有列都未更改，则无需设置此参数。默认值： ``None``。
            - **shuffle** (bool, 可选) - 是否对数据集执行shuffle。需要随机可访问的输入。默认值： ``True``，表中显示的预期顺序。
            - **drop_remainder** (bool, 可选) - 确定是否删除最后一个block，这个block的数据行数小于批处理大小。如果为 ``True``，且有更小的 `batch_size` ，可用于创建最后一个batch，那么这些行将被丢弃，而不传播到子节点。默认值： ``True``。
            - **prebatched_data** (bool, 可选) - 在创建MindSpore数据集之前生成预批处理数据。如果为 ``True``，当按索引获取每个子数据集数据时，将返回预批处理数据。否则，批处理操作将由MindSpore数据集接口：dataset.batch完成。当 `batch_size` 非常大时，建议将此选项设置为 ``True``，以提高主机上的性能。默认值： ``False``。
            - **num_parallel_workers** (int, 可选) - 并行处理数据集的工作线程（线程）数。默认值： ``1``。
            - **num_shards** (int, 可选) - 数据集将被划成的分片数。需要随机可访问的输入。指定此参数时，`num_samples` 反映每个分片的最大样本数。默认值： ``None``。
            - **shard_id** (int, 可选) - `num_shards` 内的shard ID。需要随机可访问的输入。仅当同时指定了 `num_shards` 时必须指定此参数。默认值： ``None``。
            - **python_multiprocessing** (bool, 可选) - 并行使用多处理Python函数per_batch_map和multi-processing。 如果函数计算量很大，此选项可能会很有用。默认值： ``False``。

        返回：
            BatchDataset，批处理之后的数据集。

    .. py:method:: mindelec.data.Dataset.get_columns_list()

        获取column列表。

        返回：
            list[str]，最终统一数据集的列名列表。

    .. py:method:: mindelec.data.Dataset.set_constraint_type(constraint_type='Equation')

        设置数据集的约束类型。

        参数：
            - **constraint_type** (Union[str, dict]) - 指定数据集的约束类型。如果是string，则所有子数据集的约束类型将设置为相同的类型，例如 ``"Equation"``、``"Label"`` 和 ``"Function"`` 等。如果是dict，则子数据集及其约束类型由对（键，值）指定。默认值： ``"Equation"``。
