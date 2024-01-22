"""
Calculate m2g edge & features
"""
#pylint: disable=W1203, W1202

import collections
import logging
from functools import partial
from multiprocessing import Pool, cpu_count
import os
import shutil

import numpy as np
import pandas as pd

from spherical_geometry.polygon import SphericalPolygon

from .get_grid2mesh_edge import get_number_of_parallels
from .utils import construct_abs_path, get_basic_env_info
from .get_mesh_edges import move_neg_zero, get_coor_idx, get_length, get_z_diff, get_y_diff, get_x_diff

logger = logging.getLogger()


def prepare_grid_mesh(config):
    input_path, _, _, level, resolution = get_basic_env_info(config)
    resolution_file = construct_abs_path(input_path, config["resolution_file"], level, resolution)
    mesh_file = construct_abs_path(input_path, config["mesh_node"], level, resolution)

    grid = np.load(resolution_file)
    mesh_node_array = np.load(mesh_file)
    mesh = pd.DataFrame(data=mesh_node_array, columns=['mesh_lon', 'mesh_lat']).to_numpy()
    return grid, mesh


def preprocess_mesh_polygon(mesh):
    """Preprocessing, mainly calculating from_lonlat of mesh vertex triangles"""
    triangles_polygon = collections.OrderedDict()
    for idx in range(0, len(mesh), 3):
        m1_x, m1_y = mesh[idx]
        m2_x, m2_y = mesh[idx + 1]
        m3_x, m3_y = mesh[idx + 2]
        polygon = SphericalPolygon.from_lonlat([m1_x, m2_x, m3_x], [m1_y, m2_y, m3_y])
        triangles_polygon[idx] = (polygon, [m1_x, m1_y, m2_x, m2_y, m3_x, m3_y])

    count = len(list(triangles_polygon.keys()))
    logger.info(f"preprocess mesh shape={mesh.shape}.The size of the processing result \
                triangles_polygon is {count}.")

    return triangles_polygon


#pylint: disable=W0102
def func_wrapper(long_lat, polygon, last_position=[-1,]):
    """wrapper function"""
    x, y = long_lat[0], long_lat[1]
    ret = -1
    last = last_position[0]
    # try use the last mesh
    if last in polygon.keys():
        if polygon[last][0].contains_lonlat(x, y):
            return last

    for k, v in polygon.items():
        if v[0].contains_lonlat(x, y):
            ret = k
            break

    # cannot find mesh for grid: lon={x},lat={y}, use the last mesh
    if ret == -1:
        logger.info(f"cannot find mesh for grid: lon={x},lat={y}, use the last mesh={last}")
        return last

    last_position[0] = ret
    return ret


def get_m2g_edge_file_tmp(config, idx):
    """Obtains the m2g edge storage directory and file name of the specified process ID."""
    _, _, tmp_path, level, resolution = get_basic_env_info(config)
    m2g_edge = construct_abs_path(tmp_path, config["mesh2grid_edge"], level, resolution)

    m2g_edge_dir, _ = get_m2g_cached_folder(config)
    filename = os.path.basename(m2g_edge)
    parts = filename.split('.')
    new_m2g_edge_filename = '.'.join(parts[:-1]) + '_' + str(idx) + '.' + parts[-1]
    new_m2g_edge = os.path.join(m2g_edge_dir, new_m2g_edge_filename)
    return m2g_edge_dir, new_m2g_edge


def get_m2g_edge_features_file_tmp(config, idx):
    """Obtain the temporary file for saving the m2g edge features of the current process."""
    _, _, tmp_path, level, resolution = get_basic_env_info(config)
    m2g_edge_features = construct_abs_path(tmp_path, config["mesh2grid_edge_feats"], level, resolution)

    _, m2g_edge_features_dir = get_m2g_cached_folder(config)
    filename = os.path.basename(m2g_edge_features)
    parts = filename.split('.')
    new_filename = '.'.join(parts[:-1]) + '_' + str(idx) + '.' + parts[-1]
    m2g_edge_features = os.path.join(m2g_edge_features_dir, new_filename)
    return m2g_edge_features_dir, m2g_edge_features


def get_m2g_cached_folder(config):
    """gets the name of the m2g temporary directory"""
    _, _, tmp_path, level, resolution = get_basic_env_info(config)
    m2g_edge = construct_abs_path(tmp_path, config["mesh2grid_edge"], level, resolution)
    m2g_edge_dir = os.path.join(os.path.dirname(m2g_edge), "m2g_level{}_tmp".format(level))

    m2g_edge_features = construct_abs_path(tmp_path, config["mesh2grid_edge_feats"], level, resolution)
    m2g_edge_features_dir = os.path.join(os.path.dirname(m2g_edge_features), "m2g_features_level{}_tmp".format(level))

    return m2g_edge_dir, m2g_edge_features_dir


def create_m2g_cached_folder(config):
    """Create Cached Directory"""
    m2g_edge_dir, m2g_edge_features_dir = get_m2g_cached_folder(config)
    if not os.path.exists(m2g_edge_dir):
        os.mkdir(m2g_edge_dir)

    if not os.path.exists(m2g_edge_features_dir):
        os.mkdir(m2g_edge_features_dir)


def delete_m2g_cached_folder(config):
    """Deleting Cached Directory"""
    m2g_edge_dir, m2g_edge_features_dir = get_m2g_cached_folder(config)
    if os.path.exists(m2g_edge_dir):
        shutil.rmtree(m2g_edge_dir, True)

    if os.path.exists(m2g_edge_features_dir):
        shutil.rmtree(m2g_edge_features_dir, True)


def clean_cached_dir(config):
    delete_m2g_cached_folder(config)
    create_m2g_cached_folder(config)


def generate_part_m2g_edge(config, idx, total_process, grid, polygon):
    """Calculate m2g edge"""
    _, m2g_edge_file_tmp = get_m2g_edge_file_tmp(config, idx)

    grid_split = np.array_split(grid, total_process)
    patials_grid, _ = grid_split[idx], len(grid_split[idx])
    logger.info(f"grid splitting, grid block={idx} , shape={patials_grid.shape}")

    f = np.vectorize(func_wrapper, signature='(n),()->()')
    position = f(patials_grid, polygon)
    valid_mesh_lon_lat = np.array([polygon[idx][1] for idx in position])

    m2g_edge = np.c_[patials_grid, valid_mesh_lon_lat]
    np.savetxt(m2g_edge_file_tmp, m2g_edge, delimiter=',')
    logger.info(f"The current process idx={idx} calculates the m2g edge successfully. \
                The result is saved in ={os.path.basename(m2g_edge_file_tmp)}, m2g edge's shape={m2g_edge.shape}.")
    return m2g_edge_file_tmp


def generate_mesh2grid_edge_features(config, idx, m2g_edge_file):
    """Calculate grid2mesh edge features"""
    logger.info("The current process idx = {} starts to calculate m2g edge features. "
                "The calculation input file = {}. Please wait.".format(idx, os.path.basename(m2g_edge_file)))
    edge_df = pd.read_csv(m2g_edge_file, header=None,
                          names=["grid_lon", "grid_lat", "m1_x", "m1_y", "m2_x", "m2_y", "m3_x", "m3_y"]).to_numpy()

    # Extract the m1_x, m1_y, m2_x, m2_y, m3_x, m3_y columns separately using the array slice operation.
    m1 = edge_df[:, [2, 3, 0, 1]]
    m2 = edge_df[:, [4, 5, 0, 1]]
    m3 = edge_df[:, [6, 7, 0, 1]]

    # Combine the reorganized arrays.
    edge_df = np.vstack([m1, m2, m3])
    edge_df = pd.DataFrame(edge_df, columns=['mesh_lon', 'mesh_lat', 'grid_lon', 'grid_lat'])

    edge_df = edge_df.round(3)
    edge_df = edge_df.applymap(move_neg_zero)
    edge_df = edge_df.drop_duplicates(subset=['mesh_lon', 'mesh_lat', 'grid_lon', 'grid_lat'], keep='first')
    edge_df = edge_df.reset_index(drop=True)

    edge_df['idx'] = edge_df.index
    edge_df['sender'] = edge_df.apply(lambda row: get_coor_idx(row['mesh_lon'], row['mesh_lat']), axis=1)
    edge_df['receiver'] = edge_df.apply(lambda row: get_coor_idx(row['grid_lon'], row['grid_lat']), axis=1)
    edge_df['length'] = edge_df.apply(lambda row: get_length(row['mesh_lon'], row['mesh_lat'],
                                                             row['grid_lon'], row['grid_lat']), axis=1)
    edge_df['diff_x'] = edge_df.apply(lambda row: get_x_diff(row['mesh_lon'], row['mesh_lat'],
                                                             row['grid_lon'], row['grid_lat']), axis=1)
    edge_df['diff_y'] = edge_df.apply(lambda row: get_y_diff(row['mesh_lon'], row['mesh_lat'],
                                                             row['grid_lon'], row['grid_lat']), axis=1)
    edge_df['diff_z'] = edge_df.apply(lambda row: get_z_diff(row['mesh_lon'], row['mesh_lat'],
                                                             row['grid_lon'], row['grid_lat']), axis=1)

    _, m2g_edge_feats = get_m2g_edge_features_file_tmp(config, idx)
    edge_df.to_csv(m2g_edge_feats)
    logger.info(f"The m2g edge features calculated successfully by the current process idx={idx}.\
                The result is saved in ={os.path.basename(m2g_edge_feats)}, \
                    m2g edge's shape={edge_df.shape}.")


def merge_m2g_cached_edges(config):
    """Combine the calculation results of all subprocesses. The result is m2g features."""
    _, _, tmp_path, level, resolution = get_basic_env_info(config)
    m2g_edge = construct_abs_path(tmp_path, config["mesh2grid_edge_feats"], level, resolution)

    final_pd = pd.DataFrame()
    _, m2g_edge_features_dir = get_m2g_cached_folder(config)
    for file in os.listdir(m2g_edge_features_dir):
        filename = os.path.join(m2g_edge_features_dir, file)
        cur_m2g = pd.read_csv(filename)
        final_pd = pd.concat([final_pd, cur_m2g], ignore_index=True).reset_index(drop=True)

    final_pd = final_pd.drop_duplicates(subset=['mesh_lon', 'mesh_lat', 'grid_lon', 'grid_lat'], keep='first')
    final_pd = final_pd.reset_index(drop=True)
    final_pd['idx'] = final_pd.index

    final_pd.to_csv(m2g_edge)
    delete_m2g_cached_folder(config)
    logger.info(f"The m2g edge & features are successfully generated and saved in {m2g_edge}. \
                m2g edge's shape={final_pd.shape}.")


def worker_helper(idx, total_process, config):
    """Working function of the child process"""
    grid, mesh = prepare_grid_mesh(config)
    triangles_polygon = preprocess_mesh_polygon(mesh)

    m2g_edge_file = generate_part_m2g_edge(config, idx, total_process, grid, triangles_polygon)
    generate_mesh2grid_edge_features(config, idx, m2g_edge_file)


def adjusting_parallels(parallels, config):
    """Adjust parallels"""
    config_parallels = get_number_of_parallels(config)

    grid, _ = prepare_grid_mesh(config)
    if grid.shape[0] < parallels:
        parallels = grid.shape[0]

    parallels = min(parallels, config_parallels)
    logger.info(f"The number of parallels is adjusted to={parallels}")

    return parallels


def generate_m2g_edge(config):
    """Calculate m2g edge & features"""

    # parallelism calculation
    parallelism = cpu_count()
    parallelism = adjusting_parallels(parallelism, config)

    # file cache cleanup
    clean_cached_dir(config)

    # multiprocess parallel computing
    pool = Pool(parallelism)
    process_lst = list(range(parallelism))
    patial_func = partial(worker_helper, total_process=parallelism, config=config)
    pool.map(patial_func, process_lst)
    pool.close()
    pool.join()

    merge_m2g_cached_edges(config)
