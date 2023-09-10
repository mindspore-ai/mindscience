sciai.utils.download_resource
==============================================

.. py:function:: sciai.utils.download_resource(model_name: str, is_force=False)

    为名为 `model_name` 的模型下载数据集与（或）checkpoints文件。
    如果模型配置文件中存在 `data_status` ，则数据集将会按照 `remote_data_path` 或 `model_path` 被下载。

    参数：
        - **model_name** (str) - 目标模型名称。
        - **is_force** (bool) - 是否强制下载数据集。

    异常：
        - **ValueError** - 如果 `model_name` 是不支持的模型名称。