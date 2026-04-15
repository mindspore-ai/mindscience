# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""visual"""

import datetime
import os
from functools import partial

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import animation
import cartopy.crs as ccrs
import cartopy.mpl.ticker as mticker
import numpy as np

from .tools import get_datapath_from_date

COLORS = (
    '#FFFFFF', '#E5E5E5', '#CCCCCC', '#B2B2B2', '#AD99AD', '#7A667A', '#473347', '#330066',
    '#59007F', '#7F00FF', '#007FFF', '#00CCFF', '#00FFFF', '#26E599', '#66BF26', '#BFE526',
    '#FFFF7F', '#FFFF00', '#FFD900', '#FFB000', '#FF7200', '#FF0000', '#CC0000', '#7F002C',
    '#CC3D6E', '#FF00FF', '#FF7FFF', '#FFBFFF', '#E5CCE5', '#E5E5E5')
LEVELS = [-80, -70, -60, -52, -48, -44, -40, -36, -32, -28, -24, -20, -16, -12, -8, -4, 0, 4, 8, 12, 16, 20, 24, 28,
          32, 36, 40, 44, 48, 52, 56]


def _get_grid(grid_resolution):
    '''
    Get the longitude and latitude grids of given resolution.
    '''
    lon = np.arange(0, 360, grid_resolution)
    lat = np.arange(90, -90, -grid_resolution)
    if grid_resolution == 0.25:
        lat = np.append(lat, -90)
    return lon, lat


def _get_wind_and_temperature(data_path, static_path):
    '''
    Get the wind speed and temperature of the given data.
    '''
    data = np.load(data_path)[0]
    static = np.load(static_path)
    u = data[..., -3] * static[-3, 0] + static[-3, 1]
    v = data[..., -2] * static[-2, 0] + static[-2, 1]
    t2m = data[..., -1] * static[-1, 0] + static[-1, 1]

    return u, v, t2m


def _animate(i, start_date, data_interval, file_path, static_path, lon, lat, ax, proj):
    '''
    Plot each frame of animation by given data.
    '''
    file_name, _ = get_datapath_from_date(start_date, i * data_interval)
    data_path = os.path.join(file_path, file_name)
    u, v, temperature = _get_wind_and_temperature(data_path, static_path)
    cs = ax.contourf(lon, lat, temperature - 273.15,
                     levels=LEVELS, colors=COLORS, transform=proj)
    ax.quiver(lon, lat, u, v, transform=ccrs.PlateCarree(180), regrid_shape=35, width=0.001, headaxislength=4,
              headlength=6, headwidth=3)
    return cs


def plt_wind_quiver(grid_resolution,
                    root_dir,
                    data_mode,
                    start_date=(2015, 1, 1, 0, 0, 0),
                    data_interval=6,
                    frames=20,
                    save_fig_path="./wind_quiver",
                    is_videos=False):
    """
    Plot a wind vector diagram.

    Args:
        grid_resolution (float): The resolution of grid.
        root_dir (str): The root dir of data, which include train_surface_static, train_surface, etc.
        data_mode (str): The mode of data, such as train, test, and valid.
        start_date (tuple): The begin date of the data, a tuple of year, month, day, hour, minute, second,
                            default (2015, 1, 1, 0, 0, 0).
        data_interval (int): The interval of data, default 6.
        frames (int): The frames of animation, default 20.
        save_fig_path (str): The path of picture or animation saved, default ./wind_quiver.
        is_videos (bool): Whether to draw an animation, default False.

    Note:
        Using `plt_wind_quiver` to draw wind quiver figure requires a Python >= 3.9 wich the installation of cartopy.

    Supported Platforms:
        ``Ascend`` ``CPU`` ``GPU``
    """
    start_date_time = datetime.datetime(*start_date)
    file_name, year_name = get_datapath_from_date(start_date_time, 0)
    surface_path = os.path.join(root_dir, data_mode + "_surface")
    surface_static_path = os.path.join(root_dir, data_mode + "_surface_static")
    data_path = os.path.join(surface_path, file_name)
    static_path = os.path.join(surface_static_path, year_name)
    lon, lat = _get_grid(grid_resolution)
    u, v, t2m = _get_wind_and_temperature(data_path, static_path)
    proj = ccrs.PlateCarree(central_longitude=0)

    fig = plt.figure(figsize=(9, 6), dpi=150)
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    cs = ax.contourf(lon, lat, t2m - 273.15, levels=LEVELS,
                     colors=COLORS, transform=proj)
    ax.quiver(lon, lat, u, v, transform=ccrs.PlateCarree(
        180), regrid_shape=30, width=0.001, headaxislength=4, headlength=6, headwidth=3)
    ax.coastlines(color='black')
    ax.set_global()

    cbar = fig.colorbar(cs, orientation='horizontal',
                        pad=0.05, aspect=20, shrink=1.2)
    cbar.set_label('temperature(celsius)')
    cbar.locator = ticker.MaxNLocator(nbins=31)
    cbar.set_ticks(LEVELS)
    cbar.update_ticks()

    xticks = [-180, -120, -60, 0, 60, 120, 180]
    ax.set_xticks(xticks, crs=proj)
    ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=proj)
    lon_formatter = mticker.LongitudeFormatter(zero_direction_label=True)
    lat_formatter = mticker.LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    if is_videos:
        anim = animation.FuncAnimation(fig,
                                       partial(_animate,
                                               start_date=start_date_time,
                                               data_interval=data_interval,
                                               file_path=surface_path,
                                               static_path=static_path,
                                               lon=lon,
                                               lat=lat,
                                               ax=ax,
                                               proj=proj),
                                       frames=frames,
                                       interval=1000,
                                       repeat=False)
        out_file = os.path.join(save_fig_path, "wind_quiver.mp4")
        anim.save(out_file, writer=animation.FFMpegWriter())
    else:
        out_file = os.path.join(save_fig_path, "wind_quiver.png")
        plt.savefig(out_file, dpi=150)
    print(f"Save {out_file} finished!")
