mindelec.data.ExistedDataset
============================

.. py:class:: mindelec.data.ExistedDataset(name=None, data_dir=None, columns_list=None, data_format='npy', constraint_type='Label', random_merge=True, data_config=None)

    使用给定的数据路径创建数据集。

    .. note::
        目前支持 `npy` 数据格式。

    参数：
        - **name** (str, 可选) - 指定数据集的名称。默认值： ``None``。如果 `data_config` 为 ``None``，则 `name` 应不为 ``None``。
        - **data_dir** (Union[str, list, tuple], 可选) - 已存在数据文件的路径。默认值： ``None``。如果 `data_config` 为 ``None``， `data_dir` 不应为 ``None``。
        - **columns_list** (Union[str, list, tuple], 可选) - 数据集的列名列表。默认值： ``None``。如果 `data_config` 为 ``None``， `columns_list` 不应为 ``None``。
        - **data_format** (str, 可选) - 现有数据文件的格式。默认值： ``'npy'``。
        - **constraint_type** (str, 可选) - 指定创建的数据集的约束类型。默认值： ``"Label"``。
        - **random_merge** (bool, 可选) - 指定是否随机合并给定的数据集。默认值： ``True``。
        - **data_config** (ExistedDataConfig, 可选) - ExistedDataConfig实例，收集上述信息。默认值： ``None``。如果非 ``None``，则将通过使用它来简化创建数据集类。如果为 ``None``，则(`name`, `data_dir`, `columns_list`, `data_format`, `constraint_type`, `random_merge`)的信息用于替换。

    异常：
        - **ValueError** - 当 `data_config` 为None时，参数 `name` / `data_dir` / `columns_list` 为 ``None``。
        - **TypeError** - 如果 `data_config` 不是ExistedDataConfig的实例。
        - **ValueError** - 如果 `data_format` 不是 ``'npy'``。
