mindearth.utils.plt_global_field_data
==============================================

.. py:function:: mindearth.utils.plt_global_field_data(data, feature_name, std, mean, fig_title, is_surface=False, is_error=False)

    绘制全球领域气象数据图。

    参数：
        - **data** (numpy.array) - 全球领域数据点。
        - **feature_name** (str) - 将要进行可视化的特征名。
        - **std** (numpy.array) - 每一等级变量的标准差。
        - **mean** (numpy.array) - 每一等级变量的均值。
        - **fig_title** (str) - 图像名称。
        - **is_surface** (bool) - 是否是表面特征，默认值： ``False`` 。
        - **is_error** (bool) - 是否是绘制误差，默认值： ``False`` 。