"""
Calculate g2m edge & features
"""
#pylint: disable=W1203, W1202

import logging
import os
import shutil
import time
import math
from functools import partial
from multiprocessing import Pool

import pandas as pd
import numpy as np

from .utils import construct_abs_path, get_basic_env_info
from .get_mesh_edges import move_neg_zero, get_coor_idx, get_length, get_z_diff, get_y_diff, get_x_diff, \
    coordinate_transformation, max_mesh_edge

logger = logging.getLogger()


def calculate_3d_coordinate_length(row, x2, y2, z2):
    """Directly use x/y/z 3D coordinate calculation"""
    x1, y1, z1 = row[2], row[3], row[4]
    length = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
    return length


def preprocess_grid_cartesian(config):
    """Converts the longitude and latitude of a grid into a three-dimensional coordinate axis."""
    input_path, _, tmp_path, level, resolution = get_basic_env_info(config)
    xyz_file = construct_abs_path(tmp_path, config["long_lat_with_xyz"], level, resolution)
    resolution_file = construct_abs_path(input_path, config["resolution_file"], level, resolution)

    resolution_array = np.load(resolution_file)
    grid = pd.DataFrame(data=resolution_array, columns=['longitude', 'latitude'])
    grid_xyz = grid.apply(lambda x: coordinate_transformation(x["longitude"], x["latitude"]), axis=1)
    grid_xyz = pd.DataFrame(grid_xyz.values)
    grid_xyz[['x', 'y', 'z']] = grid_xyz.iloc[:, 0].apply(pd.Series)
    grid_xyz = grid_xyz.drop(0, axis=1)

    grid_xyz = pd.concat([grid, grid_xyz], axis=1)
    grid_xyz.to_csv(xyz_file)
    logger.info(f"The main process preprocesses grid data and saves the result in ={xyz_file}. \
                grid_xyz's shape={grid_xyz.shape}.")
    return grid_xyz


def preprocess_mesh_cartesian(config):
    """Converts the longitude and latitude of a mesh into a three-dimensional coordinate axis."""
    input_path, _, tmp_path, level, resolution = get_basic_env_info(config)
    mesh_node_file = construct_abs_path(input_path, config["mesh_node"], level, resolution)
    xyz_file = construct_abs_path(tmp_path, config["mesh_node_with_xyz"], level, resolution)

    mesh_node_array = np.load(mesh_node_file)
    mesh = pd.DataFrame(data=mesh_node_array, columns=['mesh_lon', 'mesh_lat'])
    mesh_xyz = mesh.apply(lambda x: coordinate_transformation(x["mesh_lon"], x["mesh_lat"]), axis=1)
    mesh_xyz = pd.DataFrame(mesh_xyz.values)
    mesh_xyz[['x', 'y', 'z']] = mesh_xyz.iloc[:, 0].apply(pd.Series)
    mesh_xyz = mesh_xyz.drop(0, axis=1)
    mesh_xyz = pd.concat([mesh, mesh_xyz], axis=1)

    mesh_xyz.to_csv(xyz_file)
    logger.info("The main process preprocesses mesh data and saves the result in ={}, "
                "mesh_xyz's shape={}.".format(xyz_file, mesh_xyz.shape))
    return mesh_xyz


def preprocess_grid_mesh_cartesian(config):
    """preprocesses and mainly calculating the cartesian."""
    grid_xyz = preprocess_grid_cartesian(config)
    mesh_xyz = preprocess_mesh_cartesian(config)
    return grid_xyz, mesh_xyz


def get_g2m_cached_folder(config):
    """gets the name of the g2m temporary directory"""
    _, _, tmp_path, level, resolution = get_basic_env_info(config)
    g2m_edge = construct_abs_path(tmp_path, config["grid2mesh_edge"], level, resolution)
    g2m_edge_dir = os.path.join(os.path.dirname(g2m_edge), "g2m_level{}_tmp".format(level))

    g2m_edge_features = construct_abs_path(tmp_path, config["grid2mesh_edge_features"], level, resolution)
    g2m_edge_features_dir = os.path.join(os.path.dirname(g2m_edge_features), "g2m_features_level{}_tmp".format(level))
    return g2m_edge_dir, g2m_edge_features_dir


def create_g2m_cached_folder(config):
    """Create Cached Directory"""
    g2m_edge_dir, g2m_edge_features_dir = get_g2m_cached_folder(config)
    if not os.path.exists(g2m_edge_dir):
        os.mkdir(g2m_edge_dir)

    if not os.path.exists(g2m_edge_features_dir):
        os.mkdir(g2m_edge_features_dir)


def delete_g2m_cached_folder(config):
    """Deleting Cached Directory"""
    g2m_edge_dir, g2m_edge_features_dir = get_g2m_cached_folder(config)
    if os.path.exists(g2m_edge_dir):
        shutil.rmtree(g2m_edge_dir, True)

    if os.path.exists(g2m_edge_features_dir):
        shutil.rmtree(g2m_edge_features_dir, True)


def clean_cached_dir(config):
    delete_g2m_cached_folder(config)
    create_g2m_cached_folder(config)


def get_g2m_edge_file_tmp(config, idx):
    """Obtains the g2m edge storage directory and file name of the specified process ID."""
    _, _, tmp_path, level, resolution = get_basic_env_info(config)
    grid2mesh_edge = construct_abs_path(tmp_path, config["grid2mesh_edge"], level, resolution)
    g2m_edge_dir, _ = get_g2m_cached_folder(config)
    filename = os.path.basename(grid2mesh_edge)
    parts = filename.split('.')
    new_g2m_edge_filename = '.'.join(parts[:-1]) + '_' + str(idx) + '.' + parts[-1]
    grid2mesh_edge = os.path.join(g2m_edge_dir, new_g2m_edge_filename)

    return g2m_edge_dir, grid2mesh_edge


def get_g2m_edge_features_file_tmp(config, idx):
    """Obtain the temporary file for saving the g2m edge features of the current process."""
    _, _, tmp_path, level, resolution = get_basic_env_info(config)
    g2m_edge_features = construct_abs_path(tmp_path, config["grid2mesh_edge_features"], level, resolution)

    _, g2m_edge_features_dir = get_g2m_cached_folder(config)
    filename = os.path.basename(g2m_edge_features)
    parts = filename.split('.')
    new_filename = '.'.join(parts[:-1]) + '_' + str(idx) + '.' + parts[-1]
    g2m_edge_features = os.path.join(g2m_edge_features_dir, new_filename)

    return g2m_edge_features_dir, g2m_edge_features


def generate_part_g2m_edge(config, idx, total_process):
    """Calculate g2m edge"""
    _, _, tmp_path, level, resolution = get_basic_env_info(config)
    grid_xyz = construct_abs_path(tmp_path, config["long_lat_with_xyz"], level, resolution)
    mesh_xyz = construct_abs_path(tmp_path, config["mesh_node_with_xyz"], level, resolution)
    _, grid2mesh_edge = get_g2m_edge_file_tmp(config, idx)

    grid = pd.read_csv(grid_xyz, header=0, usecols=["longitude", "latitude", 'x', 'y', 'z'])
    mesh = pd.read_csv(mesh_xyz, header=0, usecols=['mesh_lon', 'mesh_lat', 'x', 'y', 'z'])
    # cal current level max_len(all mesh edge)
    max_len = max_mesh_edge(config, level)
    max_len = max_len * 0.6

    grid = grid.to_numpy()
    mesh_split = np.array_split(mesh.to_numpy(), total_process)
    cur_mesh = mesh_split[idx]
    logger.info("Current process idx={}, processing mesh size={}".format(idx, cur_mesh.shape))

    patials_g2m_edge = np.empty(shape=(0, 8))
    # for each edge of g2m, calculate distance, if distance < max edge len
    for index, row in enumerate(cur_mesh):
        s_time = time.time()
        mesh_long, mesh_lat = row[0], row[1]
        mesh_x, mesh_y, mesh_z = row[2], row[3], row[4]

        f = np.vectorize(calculate_3d_coordinate_length, signature='(n),(),(),()->()')
        dist = f(grid, mesh_x, mesh_y, mesh_z)

        # after filter => [grid_lon, grid_lat, grid_x, grid_y, gird_y, dist]
        grid = np.c_[grid, dist.T]
        dist_colum_idx = 5
        res = grid[np.where(grid[:, dist_colum_idx] < max_len)]

        # patials_g2m_edge = [grid_lon, grid_lat, grid_x, grid_y, gird_y, dist, mesh_lon, mesh_lat]
        long = np.repeat(mesh_long, len(res))
        lat = np.repeat(mesh_lat, len(res))
        res = np.c_[res, long.T, lat.T]
        patials_g2m_edge = np.r_[patials_g2m_edge, res]

        # Restore the initial grid status.
        grid = np.delete(grid, [dist_colum_idx], axis=1)

        e_time = time.time()
        per_time = e_time - s_time
        total_time = per_time * len(cur_mesh)
        left_time = (len(cur_mesh) - index) * per_time
        logger.info(f"The current process ID is {idx}. It takes {per_time:.2f} seconds to process the mesh. \
                    The total time is {total_time:.2f} seconds. The remaining time is {left_time} seconds.")

    np.savetxt(grid2mesh_edge, patials_g2m_edge, delimiter=',')
    logger.info(f"The current process idx={idx} calculates the g2m edge successfully. \
                The result is saved in ={os.path.basename(grid2mesh_edge)}, g2m edge's shape={patials_g2m_edge.shape}.")
    return grid2mesh_edge


def generate_part_g2m_edge_features(config, idx, g2m_edge_file):
    """Calculate grid2mesh edge features"""
    logger.info(f"The current process idx = {idx} starts to calculate g2m edge features. \
                The calculation input file = {os.path.basename(g2m_edge_file)}. Please wait.")

    edge_df = pd.read_csv(g2m_edge_file, header=None,
                          names=["grid_lon", "grid_lat", "grid_x", "grid_y", "grid_z", "dist", "mesh_lon", "mesh_lat"])
    edge_df = edge_df.round(3)
    edge_df = edge_df.applymap(move_neg_zero)
    edge_df = edge_df.drop_duplicates(subset=["grid_lon", "grid_lat", "mesh_lon", "mesh_lat"], keep='first')
    edge_df = edge_df.reset_index(drop=True)

    edge_df['idx'] = edge_df.index
    edge_df['sender'] = edge_df.apply(lambda row: get_coor_idx(row['grid_lon'], row['grid_lat']), axis=1)
    edge_df['receiver'] = edge_df.apply(lambda row: get_coor_idx(row['mesh_lon'], row['mesh_lat']), axis=1)
    edge_df['length'] = edge_df.apply(lambda row: get_length(row['grid_lon'], row['grid_lat'],
                                                             row['mesh_lon'], row['mesh_lat']), axis=1)
    edge_df['diff_x'] = edge_df.apply(lambda row: get_x_diff(row['grid_lon'], row['grid_lat'],
                                                             row['mesh_lon'], row['mesh_lat']), axis=1)
    edge_df['diff_y'] = edge_df.apply(lambda row: get_y_diff(row['grid_lon'], row['grid_lat'],
                                                             row['mesh_lon'], row['mesh_lat']), axis=1)
    edge_df['diff_z'] = edge_df.apply(lambda row: get_z_diff(row['grid_lon'], row['grid_lat'],
                                                             row['mesh_lon'], row['mesh_lat']), axis=1)

    _, g2m_edge_feats = get_g2m_edge_features_file_tmp(config, idx)
    edge_df.to_csv(g2m_edge_feats)
    logger.info(f"The g2m edge features calculated successfully by the current process idx={idx}.\
                The result is saved in ={os.path.basename(g2m_edge_feats)}, g2m edge's shape={edge_df.shape}.")


def get_number_of_parallels(config):
    enabled = config["parallel_configuration_enabled"]
    if not enabled:
        config_parallels = 1
    else:
        config_parallels = config["number_of_parallels"]
    return config_parallels


def adjusting_parallels(parallels, mesh, config_parallels):
    if mesh.shape[0] < parallels:
        parallels = mesh.shape[0]

    parallels = min(parallels, config_parallels)
    logger.info("The number of parallels is adjusted to={}".format(parallels))
    return parallels


def worker_helper(idx, total_process, config):
    """Working function of the child process"""
    g2m_edge_file = generate_part_g2m_edge(config, idx, total_process)
    generate_part_g2m_edge_features(config, idx, g2m_edge_file)


def merge_g2m_cached_edges(config):
    """Combine the calculation results of all subprocesses. The result is g2m features."""
    _, _, tmp_path, level, resolution = get_basic_env_info(config)
    g2m_edge = construct_abs_path(tmp_path, config["grid2mesh_edge_features"], level, resolution)

    final_pd = pd.DataFrame()
    _, g2m_edge_features_dir = get_g2m_cached_folder(config)
    for file in os.listdir(g2m_edge_features_dir):
        edge_df = pd.read_csv(os.path.join(g2m_edge_features_dir, file))
        final_pd = pd.concat([final_pd, edge_df], ignore_index=True).reset_index(drop=True)

    final_pd = final_pd.drop_duplicates(subset=["grid_lon", "grid_lat", "mesh_lon", "mesh_lat"], keep='first')
    final_pd = final_pd.reset_index(drop=True)
    final_pd["idx"] = final_pd.index

    final_pd.to_csv(g2m_edge)
    delete_g2m_cached_folder(config)
    logger.info(f"The g2m edge & features are successfully generated and saved in {g2m_edge}. \
                g2m edge's shape={final_pd.shape}.")


def generate_g2m_edge(config):
    """Calculate g2m edge & features"""

    # data preprocessing and parallelism calculation
    _, mesh_xyz = preprocess_grid_mesh_cartesian(config)
    parallelism = os.cpu_count()
    config_parallels = get_number_of_parallels(config)
    parallelism = adjusting_parallels(parallelism, mesh_xyz, config_parallels)

    # file cache cleanup
    clean_cached_dir(config)

    # multiprocess parallel computing
    pool = Pool(parallelism)
    process_lst = list(range(parallelism))
    patial_func = partial(worker_helper, total_process=parallelism, config=config)
    pool.map(patial_func, process_lst)
    pool.close()
    pool.join()

    merge_g2m_cached_edges(config)
