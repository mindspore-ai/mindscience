"""
Calculate mesh node features.
"""
#pylint: disable=W1203, W1202

import logging
import math
import os

import numpy as np
import pandas as pd

from .utils import construct_abs_path, get_basic_env_info

logger = logging.getLogger()


def get_coor_idx(x, y):
    x = '%.3f' % x
    y = '%.3f' % y
    out = x + "_" + y
    return out


def move_neg_zero(x):
    # -0 -> 0
    if str(x) == "-0.0":
        return -x
    return x


def generate_mesh_node(config):
    """Calculate mesh node features."""
    input_path, _, tmp_path, level, resolution = get_basic_env_info(config)
    mesh_node_file = construct_abs_path(input_path, config["mesh_node"], level, resolution)
    features_file = construct_abs_path(tmp_path, config["mesh_node_features"], level, resolution)
    features_file_npy = construct_abs_path(tmp_path, config["mesh_node_features_npy"], level, resolution)

    logger.info(f"input file={mesh_node_file}")
    if not os.path.exists(mesh_node_file):
        logger.error(f"input file={mesh_node_file} isn't exist.")
        return

    mesh_node_array = np.load(mesh_node_file)
    df = pd.DataFrame(data=mesh_node_array, columns=['mesh_lon', 'mesh_lat'])
    df = df.round(3)
    df = df.applymap(move_neg_zero)
    df = df.drop_duplicates(subset=['mesh_lon', 'mesh_lat'], keep='first')
    df = df.reset_index(drop=True)

    # construct features
    df['idx'] = df.index
    df['cos_lat'] = df['mesh_lat'].map(lambda x: math.cos(x * math.pi / 180.0))
    df['sin_lon'] = df['mesh_lon'].map(lambda x: math.sin(x * math.pi / 180.0))
    df['cos_lon'] = df['mesh_lon'].map(lambda x: math.cos(x * math.pi / 180.0))
    df['coor'] = df.apply(lambda row: get_coor_idx(row['mesh_lon'], row['mesh_lat']), axis=1)
    df.to_csv(features_file)
    output_array = df[['cos_lat', 'sin_lon', 'cos_lon']].to_numpy()
    np.save(features_file_npy, output_array)
    logger.info(f"The mesh node features, level={level}, are calculated successfully. \
                The result is saved in {features_file}. \
                The npy file is saved in = {features_file_npy}. \
                mesh node's shape={df.shape}, npy file's shape={output_array.shape}")
