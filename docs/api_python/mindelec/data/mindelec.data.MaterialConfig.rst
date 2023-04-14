mindelec.data.MaterialConfig
=============================

.. py:class:: mindelec.data.MaterialConfig(json_file, material_dir, physical_field, customize_physical_field=None, remove_vacuum=True)

    PointCloud-Tensor生成的材料属性值，影响材料求解阶段。

    参数：
        - **json_file** (str) - 每个子模型json文件路径的物料信息。
        - **material_dir** (str) - 所有材料的目录路径，每种材料的物理量信息都各自被记录在一个文本文件中。
        - **physical_field** (dict) - Maxwell方程关注的标准物理量属性，材料解决阶段将处理这些标准物理字段。键为物理量名称，值为此物理量的默认值。
        - **customize_physical_field** (dict, 可选) - 用户可以根据其需求指定物理属性。同样，材料求解阶段也会关注它们。默认值： ``None``。
        - **remove_vacuum** (bool, 可选) - 是否删除材料属性为真空的子实体。默认值： ``True``。

    异常：
        - **TypeError** - 如果 `json_file` 不是str。
        - **TypeError** - 如果 `material_dir` 不是str。
        - **TypeError** - 如果 `physical_field` 不是dict。
        - **TypeError** - 如果 `customize_physical_field` 不是dict。
        - **TypeError** - 如果 `remove_vacuum` 不是bool。
