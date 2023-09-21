mindearth.utils.plt_global_field_data
==============================================

.. py:function:: mindearth.utils.plt_global_field_data(data, feature_name, std, mean, fig_title, is_surface=False, is_error=False)

    绘制全球领域气象数据图。

    参数：
        - **data** (numpy.array) - 全球领域数据点。
        - **feature_name** (int) - 将要进行可视化的特征名。
        - **std** (int) - 每一等级变量的标准差。
        - **mean** (int) - 每一等级变量的均值。
        - **fig_title** (int) - 图像名称。
        - **is_surface** (bool): 是否是表面特征，默认值：False。
        - **is_error** (bool): 是否是绘制误差，默认值：False。