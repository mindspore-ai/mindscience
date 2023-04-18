mindflow.data.ExistedDataConfig
================================

.. py:class:: mindflow.data.ExistedDataConfig(name, data_dir, columns_list, data_format="npy", constraint_type="Label", random_merge=True)

    设置ExistedDataset的参数。

    参数：
        - **name** (str) - 指定数据集的名称。
        - **data_dir** (Union[str, list, tuple]) - 已存在数据文件的路径。
        - **columns_list** (Union[str, list, tuple]) - 数据集的列名列表。
        - **data_format** (str, 可选) - 现有数据文件的格式，默认值： ``"npy"``。目前支持 ``"npy"`` 的格式。
        - **constraint_type** (str, 可选) - 指定创建的数据集的约束类型，默认值： ``"Label"``。
        - **random_merge** (bool, 可选) - 指定是否随机合并给定数据集，默认值： ``True``。
