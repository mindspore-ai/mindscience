"""
Associate the edges of m2g with the vertices of grid and mesh.
"""
#pylint: disable=W1203, W1202

import logging

import pandas as pd
import numpy as np

from .utils import construct_abs_path, get_basic_env_info

logger = logging.getLogger()


def union_m2g(config):
    """Associate the edges of m2g with the vertices of grid and mesh."""
    _, _, tmp_path, level, resolution = get_basic_env_info(config)
    grid_features = construct_abs_path(tmp_path, config["long_lat_features"], level, resolution)
    mesh_features = construct_abs_path(tmp_path, config["mesh_node_features"], level, resolution)
    m2g_features = construct_abs_path(tmp_path, config["mesh2grid_edge_feats"], level, resolution)
    m2g_sender = construct_abs_path(tmp_path, config["m2g_sender"], level, resolution)
    m2g_sender_npy = construct_abs_path(tmp_path, config["m2g_sender_npy"], level, resolution)
    m2g_receiver = construct_abs_path(tmp_path, config["m2g_receiver"], level, resolution)
    m2g_receiver_npy = construct_abs_path(tmp_path, config["m2g_receiver_npy"], level, resolution)

    mesh_df = pd.read_csv(mesh_features, header=0, usecols=["idx", "mesh_lon", "mesh_lat", "coor"], index_col="coor")
    m2g_df = pd.read_csv(m2g_features, header=0,
                         usecols=["idx", "mesh_lon", "mesh_lat", "grid_lon", "grid_lat", "sender", "receiver"])
    sender_join = m2g_df.join(mesh_df, lsuffix='_l', rsuffix='_r', on="sender")
    del sender_join['mesh_lon_r']
    del sender_join['mesh_lat_r']
    sender_join.rename(columns={'idx_l': 'idx', 'mesh_lon_l': 'mesh_lon',
                                'mesh_lat_l': 'mesh_lat', 'idx_r': 'idx_mesh'}, inplace=True)
    sender_join.to_csv(m2g_sender)
    m2g_sender_idx = sender_join['idx_mesh'].to_numpy()
    np.save(m2g_sender_npy, m2g_sender_idx)
    logger.info("The union m2g is successful. The sender csv result is saved in ={}, shape={};"
                "npy results are saved in ={}, shape={}".format(m2g_sender, sender_join.shape,
                                                                m2g_sender_npy, m2g_sender_idx.shape))

    grid_df = pd.read_csv(grid_features, header=0, usecols=["idx", "longitude", "latitude", "coor"], index_col="coor")
    receiver_join = m2g_df.join(grid_df, lsuffix='_l', rsuffix='_r', on="receiver")
    del receiver_join['longitude']
    del receiver_join['latitude']
    receiver_join.rename(columns={'idx_l': 'idx', 'idx_r': 'idx_grid'}, inplace=True)

    receiver_join.to_csv(m2g_receiver)
    m2g_receiver_idx = receiver_join['idx_grid'].to_numpy()
    np.save(m2g_receiver_npy, m2g_receiver_idx)
    logger.info(f"union m2g is successful. The receiver result is saved in ={m2g_receiver}, \
                shape={receiver_join.shape}. \
                    The npy result is saved in ={m2g_receiver_npy}, \
                        shape={m2g_receiver_idx.shape}.")
