"""Get grid node"""
#pylint: disable=W1203, W1202

import logging
import math
import decimal

import pandas as pd
import numpy as np

from .utils import construct_abs_path, get_basic_env_info

decimal.getcontext().rounding = "ROUND_HALF_UP"

logger = logging.getLogger()


def decimal_float(x):
    dec = decimal.Decimal
    out = dec(x).quantize(dec('0.000'))
    return float(out)


def get_coor_idx(x, y):
    x = '%.3f' % x
    y = '%.3f' % y
    out = x + "_" + y
    return out


def generate_grid_node(config):
    """Calculate the grid's longitude and latitude features."""
    input_path, _, tmp_path, level, resolution = get_basic_env_info(config)
    grid_features = construct_abs_path(tmp_path, config["long_lat_features"], level, resolution)
    grid_features_npy = construct_abs_path(tmp_path, config["long_lat_features_npy"], level, resolution)
    resolution_file = construct_abs_path(input_path, config["resolution_file"], level, resolution)

    resolution_array = np.load(resolution_file)
    df = pd.DataFrame(data=resolution_array, columns=['longitude', 'latitude'])
    df = df.round(3)

    # construct features
    df['cos_lat'] = df['latitude'].map(lambda x: math.cos(x * math.pi / 180.0))
    df['sin_lon'] = df['longitude'].map(lambda x: math.sin(x * math.pi / 180.0))
    df['cos_lon'] = df['longitude'].map(lambda x: math.cos(x * math.pi / 180.0))
    df['coor'] = df.apply(lambda row: get_coor_idx(row['longitude'], row['latitude']), axis=1)
    df.index.names = ['idx']

    df.to_csv(grid_features)
    cols_to_convert = ['cos_lat', 'sin_lon', 'cos_lon']
    np.save(grid_features_npy, df[cols_to_convert].to_numpy())
    logger.info(f"Calculate the longitude and latitude features. The CSV file is stored in ={grid_features}, \
                shape={df.shape}; and the npy file is stored in={grid_features_npy}, \
                    shape={df[cols_to_convert].to_numpy().shape}")
