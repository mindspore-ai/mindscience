"""
merge all m2m edge/m2m sender/mem receiver
"""
#pylint: disable=W1203, W1202

import logging

import numpy as np
import pandas as pd

from .utils import construct_abs_path, get_basic_env_info

logger = logging.getLogger()


def merge_all_mesh_edge(config):
    """Combine all edges on the M0-M[level] mesh"""
    merge_all_level_mesh_edge(config)
    merge_all_level_mesh_sender(config)
    merge_all_level_mesh_receiver(config)


def merge_all_level_mesh_edge(config):
    """Combine all edges on the M0-M6 mesh and save them"""
    _, _, tmp_path, level, resolution = get_basic_env_info(config)
    merged_edge_file_npy = construct_abs_path(tmp_path, config["mesh_edge_features_merged_npy"], level, resolution)
    merged_edge_file = construct_abs_path(tmp_path, config["mesh_edge_features_merged_csv"], level, resolution)

    level_features = []
    for layer in range(0, level + 1):
        mesh_edge_file = construct_abs_path(tmp_path, config["mesh_edge_features"], layer, resolution)
        level_features.append(pd.read_csv(mesh_edge_file, header=0))

    lst = [level_edge for level_edge in level_features]
    level_edge_merged = pd.concat(lst, ignore_index=True).reset_index(drop=True)
    level_edge_merged['idx'] = level_edge_merged.index

    all_edges = level_edge_merged[['length', 'diff_x', 'diff_y', 'diff_z']].to_numpy()
    np.save(merged_edge_file_npy, all_edges)
    level_edge_merged.to_csv(merged_edge_file)
    logger.info(f"The mesh edges of each layer are successfully merged. \
                The CSV result is saved in ={merged_edge_file}, \
                shape={level_edge_merged.shape}, \
                    and the Npy result is saved in ={merged_edge_file_npy}, \
                    shape={all_edges.shape}.")


def merge_all_level_mesh_sender(config):
    """Merge each sender on the M0-M6 mesh and save"""
    _, _, tmp_path, level, resolution = get_basic_env_info(config)
    merged_all_sender = construct_abs_path(tmp_path, config["mesh_merged_all_sender_csv"], level, resolution)
    merged_all_sender_npy = construct_abs_path(tmp_path, config["mesh_merged_all_sender_npy"], level, resolution)

    level_sender = []
    for layer in range(0, level + 1):
        sender_file = construct_abs_path(tmp_path, config["mesh_sender"], layer, resolution)
        level_sender.append(pd.read_csv(sender_file, header=0))

    sender_merged_df = pd.concat([sender for sender in level_sender], ignore_index=True).reset_index(drop=True)
    sender_merged_df["idx"] = sender_merged_df.index
    merged_sender_idx = sender_merged_df['idx_mesh'].to_numpy()

    sender_merged_df.to_csv(merged_all_sender)
    np.save(merged_all_sender_npy, merged_sender_idx)
    logger.info(f"The mesh senders of each layer are successfully merged. \
                The CSV result is saved in the ={merged_all_sender}, \
                shape={sender_merged_df.shape} directory. The npy result is saved in the ={merged_all_sender_npy}, \
                shape={merged_sender_idx.shape} directory.")


def merge_all_level_mesh_receiver(config):
    """Merge each receiver on the M0-M6 mesh and save"""
    _, _, tmp_path, level, resolution = get_basic_env_info(config)
    merged_all_receiver = construct_abs_path(tmp_path, config["mesh_merged_all_receiver_csv"], level, resolution)
    merged_all_receiver_npy = construct_abs_path(tmp_path, config["mesh_merged_all_receiver_npy"], level, resolution)
    level_receiver = []
    for layer in range(0, level + 1):
        receiver_file = construct_abs_path(tmp_path, config["mesh_receiver"], layer, resolution)
        level_receiver.append(pd.read_csv(receiver_file, header=0))

    receiver_merged_df = pd.concat([receiver for receiver in level_receiver], ignore_index=True).reset_index(drop=True)
    receiver_merged_df["idx"] = receiver_merged_df.index
    merged_receiver_idx = receiver_merged_df['idx_mesh'].to_numpy()

    np.save(merged_all_receiver_npy, merged_receiver_idx)
    receiver_merged_df.to_csv(merged_all_receiver)
    logger.info(f"The mesh receiver of each layer is successfully merged. \
                The result is saved in ={merged_all_receiver}, \
                shape={receiver_merged_df.shape}. The npy result is saved in ={merged_all_receiver_npy},\
                shape={merged_receiver_idx.shape}.")
