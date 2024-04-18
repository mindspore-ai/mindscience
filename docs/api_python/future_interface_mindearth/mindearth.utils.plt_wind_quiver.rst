mindearth.utils.plt_wind_quiver
==============================================

.. py:function:: mindearth.utils.plt_wind_quiver(grid_resolution, root_dir, data_mode, start_date=(2015, 1, 1, 0, 0, 0), data_interval=6, frames=20, save_fig_path="./wind_quiver", is_videos=False)

    绘制风矢图。

    参数：
        - **grid_resolution** (float): 网格分辨率。
        - **root_dir** (str): 数据的根目录，包括train_surface_static, train_surface等。
        - **data_mode** (str): 数据模式，如train，test，valid。
        - **start_date** (tuple): 数据的开始日期，年、月、日、小时、分钟、秒组成的元组。默认值（2015，1，1，0，0）。
        - **data_interval** (int): 数据间隔，默认值：6。
        - **frames** (int): 动画的帧率，默认值。
        - **save_fig_path** (str): 储存图片或动画的路径，默认值： ``"./wind_quiver"``。
        - **is_videos** (bool): 是否绘制动画，默认值：False。

    返回：
        str，数据文件名。
        str，static数据文件名。

    .. note::
        接口``plt_wind_quiver`` 要求Python版本 >= 3.9且安装cartopy三方库。