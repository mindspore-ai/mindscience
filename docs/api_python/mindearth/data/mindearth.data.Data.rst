mindearth.data.Data
=========================

.. py:class:: mindearth.data.Data(root_dir='.')

    数据处理的基类。

    参数：
        - **root_dir** (str, 可选) - 数据集的根目录文件夹路径。默认值： ``'.'``。

    异常：
        - **TypeError** - 如果 `train_dir` 的类型不是str。
        - **TypeError** - 如果 `test_dir` 的类型不是str。
