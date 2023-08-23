"""
Associate the node and edge of the mesh.
"""
#pylint: disable=W1203, W1202

import logging

import pandas as pd
from .utils import construct_abs_path, get_basic_env_info

logger = logging.getLogger()


def union_edge_grid_m2m(config):
    """
    Associate the node and edge of the mesh.
    Note: Associate the edge of each layer from M0 to M[x] with the node at the M[x] layer.
    """
    _, _, tmp_path, level, resolution = get_basic_env_info(config)

    # Node at the M[level] layer
    mesh_features = construct_abs_path(tmp_path, config["mesh_node_features"], level, resolution)
    mesh_df = pd.read_csv(mesh_features, header=0, usecols=["idx", "mesh_lon", "mesh_lat", "coor"], index_col="coor")

    for layer in range(0, level + 1):
        # M[0]-M[layer] edge of each layer
        edge_file = construct_abs_path(tmp_path, config["mesh_edge_features"], layer, resolution)
        edge_df = pd.read_csv(edge_file, header=0, usecols=["idx", "mesh_lon1", "mesh_lat1", "mesh_lon2", "mesh_lat2",
                                                            "sender", "receiver"])

        # Association of mesh edges and vertices on the sender attribute.
        sender_join = edge_df.join(mesh_df, lsuffix='_l', rsuffix='_r', on="sender")
        del sender_join['mesh_lon']
        del sender_join['mesh_lat']
        sender_join.rename(columns={'idx_l': 'idx', 'idx_r': 'idx_mesh'}, inplace=True)
        sender_file = construct_abs_path(tmp_path, config["mesh_sender"], layer, resolution)
        sender_join.to_csv(sender_file)

        # Association of mesh edges and vertices on the receiver attribute.
        receiver_join = edge_df.join(mesh_df, lsuffix='_l', rsuffix='_r', on="receiver")
        del receiver_join['mesh_lon']
        del receiver_join['mesh_lat']
        receiver_join.rename(columns={'idx_l': 'idx', 'idx_r': 'idx_mesh'}, inplace=True)
        receiver_file = construct_abs_path(tmp_path, config["mesh_receiver"], layer, resolution)
        receiver_join.to_csv(receiver_file)
        logger.info(
            f"The node and edge of the mesh are successfully associated. \
                The level is {layer}. The result is saved in {sender_file} and {receiver_file}. \
                    mesh edge's shape={edge_df.shape}, edge sender's shape={sender_join.shape}, \
                        edge receiver's shape={receiver_join.shape}")
