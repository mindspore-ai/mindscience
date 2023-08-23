"""
Associate the edges of g2m with the vertices of grid and mesh
"""
#pylint: disable=W1203, W1202

import logging

import pandas as pd
import numpy as np

from .utils import construct_abs_path, get_basic_env_info

logger = logging.getLogger()


def union_g2m(config):
    """Associate the edges of g2m with the vertices of grid and mesh."""
    _, _, tmp_path, level, resolution = get_basic_env_info(config)

    grid_features = construct_abs_path(tmp_path, config["long_lat_features"], level, resolution)
    mesh_features = construct_abs_path(tmp_path, config["mesh_node_features"], level, resolution)
    g2m_edge_features = construct_abs_path(tmp_path, config["grid2mesh_edge_features"], level, resolution)
    g2m_sender = construct_abs_path(tmp_path, config["g2m_sender"], level, resolution)
    g2m_sender_npy = construct_abs_path(tmp_path, config["g2m_sender_npy"], level, resolution)
    g2m_receiver = construct_abs_path(tmp_path, config["g2m_receiver"], level, resolution)
    g2m_receiver_npy = construct_abs_path(tmp_path, config["g2m_receiver_npy"], level, resolution)

    grid_df = pd.read_csv(grid_features, header=0, usecols=["idx", "longitude", "latitude", "coor"], index_col="coor")
    g2m_df = pd.read_csv(g2m_edge_features, header=0, usecols=["idx", "grid_lon", "grid_lat", "mesh_lon", "mesh_lat",
                                                               "sender", "receiver"])
    sender_join = g2m_df.join(grid_df, lsuffix='_l', rsuffix='_r', on="sender")

    del sender_join['longitude']
    del sender_join['latitude']
    sender_join.rename(columns={'idx_l': 'idx', 'idx_r': 'idx_grid'}, inplace=True)
    sender_join.to_csv(g2m_sender)
    g2m_sender_idx = sender_join['idx_grid'].to_numpy()
    np.save(g2m_sender_npy, g2m_sender_idx)
    logger.info("The union g2m is successful. The sender csv result is saved in ={}, shape={};."
                "npy results are saved in ={}, shape={}".format(g2m_sender, sender_join.shape,
                                                                g2m_sender_npy, g2m_sender_idx.shape))

    mesh_df = pd.read_csv(mesh_features, header=0, usecols=["idx", "mesh_lon", "mesh_lat", "coor"],
                          index_col="coor")
    receiver_join = g2m_df.join(mesh_df, lsuffix='_l', rsuffix='_r', on="receiver")
    del receiver_join['mesh_lon_r']
    del receiver_join['mesh_lat_r']
    receiver_join.rename(columns={'idx_l': 'idx', 'mesh_lon_l': 'mesh_lon',
                                  'mesh_lat_l': 'mesh_lat', 'idx_r': 'idx_mesh'}, inplace=True)
    receiver_join.to_csv(g2m_receiver)
    g2m_receiver_idx = receiver_join['idx_mesh'].to_numpy()
    np.save(g2m_receiver_npy, g2m_receiver_idx)
    logger.info(f"union g2m is successful. The receiver result is saved in ={g2m_receiver}, \
                shape={receiver_join.shape}. \
                    The npy result is saved in ={g2m_receiver_npy}, \
                        shape={g2m_receiver_idx.shape}.")
