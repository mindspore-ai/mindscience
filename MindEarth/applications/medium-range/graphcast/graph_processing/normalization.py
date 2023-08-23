"""
normalization m2m/g2m/m2g
"""
#pylint: disable=W1203, W1202

import logging

import numpy as np
import pandas as pd

from .utils import construct_abs_path, get_basic_env_info

logger = logging.getLogger()


def m2m_max_edge(config):
    """max edge of mesh to mesh"""
    _, _, tmp_path, level, resolution = get_basic_env_info(config)
    m2m_file = construct_abs_path(tmp_path, config["mesh_edge_features_merged_csv"], level, resolution)
    edge_features = pd.read_csv(m2m_file, header=0, usecols=["length"])
    return edge_features["length"].max()


def g2m_max_edge(config):
    """max edge of grid to mesh"""
    _, _, tmp_path, level, resolution = get_basic_env_info(config)
    g2m_file = construct_abs_path(tmp_path, config["grid2mesh_edge_features"], level, resolution)
    edge_features = pd.read_csv(g2m_file, header=0, usecols=["length"])
    return edge_features["length"].max()


def m2g_max_edge(config):
    """max edge of mesh to grid"""
    _, _, tmp_path, level, resolution = get_basic_env_info(config)
    m2g_file = construct_abs_path(tmp_path, config["mesh2grid_edge_feats"], level, resolution)
    edge_features = pd.read_csv(m2g_file, header=0, usecols=["length"])
    return edge_features["length"].max()


def normalize_edge(config):
    """normalize all edge"""
    normalize_m2m_edge(config)
    normalize_g2m_edge(config)
    normalize_m2g_edge(config)


def normalize_m2m_edge(config):
    """mesh2mesh edge standardization"""
    m2m_max_length = m2m_max_edge(config)

    _, _, tmp_path, level, resolution = get_basic_env_info(config)
    m2m_edge_file = construct_abs_path(tmp_path, config["mesh_edge_features_merged_npy"], level, resolution)
    m2m_edge = np.load(m2m_edge_file)
    m2m_edge = m2m_edge / m2m_max_length

    normal_file = construct_abs_path(tmp_path, config["mesh_edge_features_merged_normalization"], level, resolution)
    np.save(normal_file, m2m_edge)
    logger.info(f"Normalization of m2m succeeded, results saved in ={normal_file}, shape={m2m_edge.shape}")


def normalize_g2m_edge(config):
    """g2m edge standardization"""
    g2m_max_length = g2m_max_edge(config)

    _, _, tmp_path, level, resolution = get_basic_env_info(config)
    g2m_edge_file = construct_abs_path(tmp_path, config["grid2mesh_edge_features"], level, resolution)
    normal_file = construct_abs_path(tmp_path, config["g2m_edge_normalization"], level, resolution)

    # usecols=(13,14,15,16) ==> length,diff_x,diff_y,diff_z
    g2m_edge = np.loadtxt(g2m_edge_file, usecols=(13, 14, 15, 16), delimiter=",", skiprows=1)
    g2m_edge = g2m_edge / g2m_max_length
    np.save(normal_file, g2m_edge)
    logger.info(f"Normalization of g2m succeeded, results saved in ={normal_file}, shape={g2m_edge.shape}")


def normalize_m2g_edge(config):
    """m2g edge standardization"""
    m2g_max_length = m2g_max_edge(config)

    _, _, tmp_path, level, resolution = get_basic_env_info(config)
    m2g_edge_file = construct_abs_path(tmp_path, config["mesh2grid_edge_feats"], level, resolution)
    normal_file = construct_abs_path(tmp_path, config["m2g_edge_normalization"], level, resolution)
    # usecols=(9,10,11,12) ==> length,diff_x,diff_y,diff_z
    m2g_edge = np.loadtxt(m2g_edge_file, usecols=(9, 10, 11, 12), delimiter=",", skiprows=1)
    m2g_edge = m2g_edge / m2g_max_length
    np.save(normal_file, m2g_edge)
    logger.info(f"Normalize m2g successfully, results saved in ={normal_file}, shape={m2g_edge.shape}")
