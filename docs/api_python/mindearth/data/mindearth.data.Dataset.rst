mindearth.data.Dataset
=========================

.. py:class:: mindearth.data.Dataset(dataset_generator, distribute=False, num_workers=1, shuffle=True)

    创建训练，验证和测试的数据集，并且输出mindspore.dataset.GeneratorDataset类的实例。

    参数：
        - **dataset_generator** (Data) - 气象数据的数据生成器。
        - **distribute** (bool, 可选) - 是否对数据集执行分布式处理。默认值： ``False``。
        - **num_workers** (int, 可选) - 并行处理数据集的工作线程（线程）数。默认值： ``1``。
        - **shuffle** (bool, 可选) - 并是否对数据集执行shuffle。需要随机可访问的输入。默认值： ``True``，表中显示的预期顺序。


    .. py:method:: mindearth.data.Dataset.create_dataset(batch_size)

        创建数据集。

        参数：
            - **batch_size** (int, 可选) - 每个批处理创建的行数，int值。

        返回：
            BatchDataset，批处理之后的数据集。
