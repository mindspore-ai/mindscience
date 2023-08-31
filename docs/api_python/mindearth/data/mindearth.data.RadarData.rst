mindearth.data.RadarData
============================

.. py:class:: mindearth.data.RadarData(data_params, run_mode='train')

    RadarData类通过MindSpore框架处理Dgmr radar数据集生成数据生成器。RadarData类继承了Data类。

    参数：
        - **data_params** (dict) - 模型中的相关数据参数。
        - **run_mode** (str, 可选) - 决定数据集用于训练、验证还是测试。支持 ``'train'``,  ``'test'``,  ``'valid'``。默认值： ``'train'``。


