"""
graph processing
"""
import os

from mindearth import create_logger
from mindearth.utils import load_yaml_config

from graph_processing import generate_grid_node
from graph_processing import generate_mesh_node
from graph_processing import generate_mesh_edges
from graph_processing import union_edge_grid_m2m
from graph_processing import generate_m2g_edge
from graph_processing import generate_g2m_edge
from graph_processing import union_g2m
from graph_processing import merge_all_mesh_edge
from graph_processing import normalize_edge
from graph_processing import union_m2g
from graph_processing import copy_result2dir
from graph_processing import make_dir


if __name__ == '__main__':
    current_directory = os.path.dirname(os.path.abspath(__file__))
    logger = create_logger(path=os.path.join(current_directory, "results.log"))
    config = load_yaml_config('./graph_processing/graph_construct.yaml')
    make_dir(config)
    generate_grid_node(config)
    generate_mesh_node(config)
    generate_mesh_edges(config)
    union_edge_grid_m2m(config)
    generate_g2m_edge(config)
    generate_m2g_edge(config)
    union_g2m(config)
    union_m2g(config)
    merge_all_mesh_edge(config)
    normalize_edge(config)
    copy_result2dir(config)
