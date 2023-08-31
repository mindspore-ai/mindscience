mindearth.data.Era5Data
=========================

.. py:class:: mindearth.data.Era5Data(data_params, run_mode='train')

    Era5Data类通过MindSpore框架处理ERA5数据集生成数据生成器。Era5Data类继承了Data类。

    参数：
        - **data_params** (dict) - 模型中的相关数据参数。
        - **run_mode** (str, 可选) - 决定数据集用于训练、验证还是测试。支持 ``'train'``,  ``'test'``,  ``'valid'``。默认值： ``'train'``。


