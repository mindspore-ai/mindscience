"""
Copy the input data file of the Graphcast model to the result directory
"""
#pylint: disable=W1203, W1202

import logging
import os
import shutil

from .utils import construct_abs_path, get_basic_env_info

logger = logging.getLogger()


def mkdir_geometry_dir(config):
    """Create a directory for saving final results."""
    _, output_path, _, level, resolution = get_basic_env_info(config)
    output_path = construct_abs_path(output_path, config["geometry"], level, resolution)

    if os.path.exists(output_path):
        shutil.rmtree(output_path, True)

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    return output_path


def copy_result2dir(config):
    """Copy the input data file of the Graphcast model to the result directory."""
    source = []
    _, _, tmp_path, level, resolution = get_basic_env_info(config)
    grid_npy_file = construct_abs_path(tmp_path, config["long_lat_features_npy"], level, resolution)
    source.append(grid_npy_file)

    g2m_normal = construct_abs_path(tmp_path, config["g2m_edge_normalization"], level, resolution)
    source.append(g2m_normal)

    g2m_sender_npy = construct_abs_path(tmp_path, config["g2m_sender_npy"], level, resolution)
    g2m_receiver_npy = construct_abs_path(tmp_path, config["g2m_receiver_npy"], level, resolution)
    source.append(g2m_sender_npy)
    source.append(g2m_receiver_npy)

    m2g_normal = construct_abs_path(tmp_path, config["m2g_edge_normalization"], level, resolution)
    source.append(m2g_normal)
    m2g_sender_npy = construct_abs_path(tmp_path, config["m2g_sender_npy"], level, resolution)
    m2g_receiver_npy = construct_abs_path(tmp_path, config["m2g_receiver_npy"], level, resolution)
    source.append(m2g_sender_npy)
    source.append(m2g_receiver_npy)

    m2m_normal = construct_abs_path(tmp_path, config["mesh_edge_features_merged_normalization"], level, resolution)
    source.append(m2m_normal)

    mesh_node_npy = construct_abs_path(tmp_path, config["mesh_node_features_npy"], level, resolution)
    source.append(mesh_node_npy)
    m2m_sender_npy = construct_abs_path(tmp_path, config["mesh_merged_all_sender_npy"], level, resolution)
    m2m_receiver_npy = construct_abs_path(tmp_path, config["mesh_merged_all_receiver_npy"], level, resolution)
    source.append(m2m_sender_npy)
    source.append(m2m_receiver_npy)

    dest_dir = mkdir_geometry_dir(config)
    for file in source:
        new_file = os.path.join(dest_dir, os.path.basename(file))
        shutil.copy(file, new_file)
        logger.info(f"Files={file} are collected successfully.")
    logger.info(f"Succeeded in collecting all graphcast data preprocessing files. \
                The results are saved in = {dest_dir}.")
