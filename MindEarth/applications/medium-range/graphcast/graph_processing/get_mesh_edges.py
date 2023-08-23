"""
Calculate mesh edge features.
"""
#pylint: disable=W1203, W1202

import logging
import math
import decimal
import os

import pandas as pd
import numpy as np

from .utils import construct_abs_path, get_basic_env_info

logger = logging.getLogger()
R = 6371
decimal.getcontext().rounding = "ROUND_HALF_UP"


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


def get_cartesian(lon, lat):
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    z = R * np.sin(lat)
    return x, y, z


def coordinate_transformation(lon, lat):
    x, y, z = get_cartesian(lon, lat)
    return x, y, z


def get_length(r_lon, r_lat, s_lon, s_lat):
    x1, y1, z1 = get_cartesian(r_lon, r_lat)
    x2, y2, z2 = get_cartesian(s_lon, s_lat)
    length = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
    return length


def get_length2(r_lon, r_lat, s_lon, s_lat):
    length = math.sqrt((r_lon - s_lon) ** 2 + (r_lat - s_lat) ** 2)
    return length


def get_cartesian_x(lon, lat):
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    x = R * np.cos(lat) * np.cos(lon)
    return x


def get_cartesian_y(lon, lat):
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    y = R * np.cos(lat) * np.sin(lon)
    return y


def get_cartesian_z(lon, lat):
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    z = R * np.sin(lat)
    return z


def get_x_diff(r_lon, r_lat, s_lon, s_lat):
    x1 = get_cartesian_x(r_lon, r_lat)
    x2 = get_cartesian_x(s_lon, s_lat)
    return x2 - x1


def get_y_diff(r_lon, r_lat, s_lon, s_lat):
    y1 = get_cartesian_y(r_lon, r_lat)
    y2 = get_cartesian_y(s_lon, s_lat)
    return y2 - y1


def get_z_diff(r_lon, r_lat, s_lon, s_lat):
    z1 = get_cartesian_z(r_lon, r_lat)
    z2 = get_cartesian_z(s_lon, s_lat)
    return z2 - z1


def max_mesh_edge(config, level):
    _, _, tmp_path, level, resolution = get_basic_env_info(config)
    file = construct_abs_path(tmp_path, config["mesh_edge_features"], level, resolution)
    edge_features = pd.read_csv(file, header=0, usecols=["length"])
    length = edge_features["length"].max()
    return length


def generate_mesh_edges(config):
    """
    Calculate mesh edge features.
    Note: For the mesh edge feature processing, the edge of each layer from M0 to M[x] needs to be processed.
    """
    input_path, _, tmp_path, level, resolution = get_basic_env_info(config)
    for cur_level in range(0, level + 1):
        edge_file = construct_abs_path(input_path, config["mesh_edge"], cur_level, resolution)
        if not os.path.exists(edge_file):
            logger.error(f"input file={edge_file} isn't exist.")
            return

        mesh_edge_array = np.load(edge_file)
        edge_df = pd.DataFrame(data=mesh_edge_array, columns=["mesh_lon1", "mesh_lat1", "mesh_lon2", "mesh_lat2"])
        edge_df = edge_df.round(3)
        edge_df = edge_df.applymap(move_neg_zero)
        edge_df = edge_df.drop_duplicates(subset=["mesh_lon1", "mesh_lat1", "mesh_lon2", "mesh_lat2"], keep='first')
        edge_df = edge_df.reset_index(drop=True)

        edge_df['idx'] = edge_df.index
        edge_df['sender'] = edge_df.apply(lambda row: get_coor_idx(row['mesh_lon1'], row['mesh_lat1']), axis=1)
        edge_df['receiver'] = edge_df.apply(lambda row: get_coor_idx(row['mesh_lon2'], row['mesh_lat2']), axis=1)
        edge_df['length'] = edge_df.apply(lambda row: get_length(row['mesh_lon1'], row['mesh_lat1'],
                                                                 row['mesh_lon2'], row['mesh_lat2']), axis=1)
        edge_df['diff_x'] = edge_df.apply(lambda row: get_x_diff(row['mesh_lon1'], row['mesh_lat1'],
                                                                 row['mesh_lon2'], row['mesh_lat2']), axis=1)
        edge_df['diff_y'] = edge_df.apply(lambda row: get_y_diff(row['mesh_lon1'], row['mesh_lat1'],
                                                                 row['mesh_lon2'], row['mesh_lat2']), axis=1)
        edge_df['diff_z'] = edge_df.apply(lambda row: get_z_diff(row['mesh_lon1'], row['mesh_lat1'],
                                                                 row['mesh_lon2'], row['mesh_lat2']), axis=1)

        features_file = construct_abs_path(tmp_path, config["mesh_edge_features"], cur_level, resolution)
        edge_df.to_csv(features_file)
        logger.info(f"The mesh edge features, level={cur_level}, are calculated successfully. \
                    The result is saved in {features_file}. mesh edge's shape={edge_df.shape}")
