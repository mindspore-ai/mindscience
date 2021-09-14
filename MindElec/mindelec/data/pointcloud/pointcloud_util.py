# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""pointcloud utils"""
#pylint: disable=W0212
from importlib import import_module
import json
from time import time

from mindspore import log as logger


def json_2_dict(json_file):
    """
    Utility function to transform json to dict object

    Args:
        json_file (str): raw json data

    Returns:
         dict, python dict object which is material information for all sub-solid
    """
    with open(json_file) as f:
        material_dict = json.load(f)
    return material_dict


def load_data(file_dir, file_group_list, group_idx, shared_memory_dict):
    """
    load raw stp model data from disk, in this function we only read one group stp files (Several groups in total)

    Args:
        file_dir: directory path for stp files
        file_group_list: stp file names in current group
        group_idx: group index
        shared_memory_dict: a shared memory dictionary to record TopoDS_Solid objects from stp files
    """
    data_exchange = import_module("OCC.Extend.DataExchange")
    read_step_file = getattr(data_exchange, "read_step_file")
    group_result = []
    for _, file_name in enumerate(file_group_list):
        group_result.append(read_step_file(file_dir + file_name[0], as_compound=False))
    shared_memory_dict[group_idx] = group_result
    logger.info("Group id: {0}, group_dict_length: {1}".format(group_idx, len(group_result)))


def sections_generation(dataset, model_group, group_id, block_id, block_number):
    """
    Generate section from a list of model according to block number and block id.

    Args:
        dataset (PointCloud): PointCloud object, we extract this function from class is to use multi-processing library
        model_group (list): a list of model_info, we define model_info = tuple(TopoDS_Solid, int)
        group_id: Model list group index
        block_id: block index
        block_number: Total block number

    Returns:
        list[dict], section generation result
    """

    t_group_begin = time()
    section_info_dict = dataset._section_building_impl(model_info_list=model_group, block_idx=block_id,
                                                       block_number=block_number)

    sections_info_collection = []
    for _, section_weight_vector in section_info_dict.items():
        sections_info_collection.extend(section_weight_vector)
    t_group = time() - t_group_begin
    logger.info("#{0} group #{1} block finished all tasks, build {2} sections, time costs: {3}"
                .format(group_id, block_id, len(sections_info_collection), t_group))
    return sections_info_collection


def inner_point_generation(dataset, section_group_tasks):
    """
    Solve sampling space from a group of sections.

    Args:
        dataset (PointCloud): PointCloud object, we extract this function from class is to use multi-processing library
        section_group_tasks: a list of SectionInfo object, the task we need to solve

    Returns:
        list[dict], space solution we obtained from this module, a callback function will bring this into shared memory
    """
    t_group_begin = time()
    group_section_result = []
    for section_info in section_group_tasks:
        t_task_begin = time()

        section_result = dataset._inner_point_from_section(section_info)
        group_section_result.extend(section_result)

        t_task_end = time()
        if t_task_end - t_task_begin >= 1:
            logger.info("Model #{0}, section #{1}, analyse time costs: {2}.".
                        format(section_info.model_idx(), section_info.z_idx(), t_task_end - t_task_begin))
    logger.info("Group tasks finished, Time costs: {}".format(time() - t_group_begin))
    return group_section_result


class Topology:
    """
    Topology traversal
    implements topology traversal from any TopoDS_Shape
    this class lets you find how various topological entities are connected from one to another
    find the faces connected to an edge, find the vertices this edge is made from, get all faces connected to
    a vertex, and find out how many topological elements are connected from a source
    *note* when traversing TopoDS_Wire entities, its advised to use the specialized
    ``WireExplorer`` class, which will return the vertices / edges in the expected order

    Args:
        my_shape (TopoDS_*): the shape which topology will be traversed
        ignore_orientation (bool): filter out TopoDS_* entities of similar TShape but different Orientation
        for instance, a cube has 24 edges, 4 edges for each of 6 faces
        that results in 48 vertices, while there are only 8 vertices that have a unique
        geometric coordinate
        in certain cases ( computing a graph from the topology ) its preferable to return
        topological entities that share similar geometry, though differ in orientation
        by setting the ``ignore_orientation`` variable
        to True, in case of a cube, just 12 edges and only 8 vertices will be returned
        for further reference see TopoDS_Shape IsEqual / IsSame methods
    """
    def __init__(self, my_shape, ignore_orientation=False):

        self.my_shape = my_shape
        self.ignore_orientation = ignore_orientation

        top_abs = import_module("OCC.Core.TopAbs")
        top_abs_vertex = getattr(top_abs, "TopAbs_VERTEX")
        top_abs_edge = getattr(top_abs, "TopAbs_EDGE")
        top_abs_face = getattr(top_abs, "TopAbs_FACE")
        top_abs_wire = getattr(top_abs, "TopAbs_WIRE")
        top_abs_shell = getattr(top_abs, "TopAbs_SHELL")
        top_abs_solid = getattr(top_abs, "TopAbs_SOLID")
        top_abs_compound = getattr(top_abs, "TopAbs_COMPOUND")
        top_abs_compsolid = getattr(top_abs, "TopAbs_COMPSOLID")

        top_ods = import_module("OCC.Core.TopoDS")
        top_ods = getattr(top_ods, "topods")

        self.topology_factory = {
            top_abs_vertex: top_ods.Vertex,
            top_abs_edge: top_ods.Edge,
            top_abs_face: top_ods.Face,
            top_abs_wire: top_ods.Wire,
            top_abs_shell: top_ods.Shell,
            top_abs_solid: top_ods.Solid,
            top_abs_compound: top_ods.Compound,
            top_abs_compsolid: top_ods.CompSolid
        }

    def _loop_topology(self, topology_type, topological_entity=None, topology_type_to_avoid=None):
        """
        this could be a faces generator for a python TopoShape class
        that way you can just do:
        for face in srf.faces:
            processFace(face)
        """
        top_abs = import_module("OCC.Core.TopAbs")
        top_abs_vertex = getattr(top_abs, "TopAbs_VERTEX")
        top_abs_edge = getattr(top_abs, "TopAbs_EDGE")
        top_abs_face = getattr(top_abs, "TopAbs_FACE")
        top_abs_wire = getattr(top_abs, "TopAbs_WIRE")
        top_abs_shell = getattr(top_abs, "TopAbs_SHELL")
        top_abs_solid = getattr(top_abs, "TopAbs_SOLID")
        top_abs_compound = getattr(top_abs, "TopAbs_COMPOUND")
        top_abs_compsolid = getattr(top_abs, "TopAbs_COMPSOLID")

        top_ods = import_module("OCC.Core.TopoDS")
        top_ods_wire = getattr(top_ods, "TopoDS_Wire")
        top_ods_vertex = getattr(top_ods, "TopoDS_Vertex")
        top_ods_edge = getattr(top_ods, "TopoDS_Edge")
        top_ods_face = getattr(top_ods, "TopoDS_Face")
        top_ods_shell = getattr(top_ods, "TopoDS_Shell")
        top_ods_solid = getattr(top_ods, "TopoDS_Solid")
        top_ods_compound = getattr(top_ods, "TopoDS_Compound")
        top_ods_compsolid = getattr(top_ods, "TopoDS_CompSolid")

        topology_types_dict = {
            top_abs_vertex: top_ods_vertex,
            top_abs_edge: top_ods_edge,
            top_abs_face: top_ods_face,
            top_abs_wire: top_ods_wire,
            top_abs_shell: top_ods_shell,
            top_abs_solid: top_ods_solid,
            top_abs_compound: top_ods_compound,
            top_abs_compsolid: top_ods_compsolid
        }

        if topology_type not in topology_types_dict.keys():
            raise RuntimeError("Invalid topology type: {}".format(topology_type))

        top_exp = import_module("OCC.Core.TopExp")
        top_exp_explorer = getattr(top_exp, "TopExp_Explorer")
        self.top_exp = top_exp_explorer()

        if topological_entity is None and topology_type_to_avoid is None:
            self.top_exp.Init(self.my_shape, topology_type)
        elif topological_entity is None and topology_type_to_avoid is not None:
            self.top_exp.Init(self.my_shape, topology_type, topology_type_to_avoid)
        elif topology_type_to_avoid is None:
            self.top_exp.Init(topological_entity, topology_type)
        elif topology_type_to_avoid:
            self.top_exp.Init(topological_entity, topology_type, topology_type_to_avoid)
        seq = []
        hashes = []  # list that stores hashes to avoid redundancy

        top_tools = import_module("OCC.Core.TopTools")
        top_tools_list_of_shape = getattr(top_tools, "TopTools_ListOfShape")
        top_tools_list_iterator_of_list_of_shape = getattr(top_tools, "TopTools_ListIteratorOfListOfShape")

        occ_seq = top_tools_list_of_shape()
        while self.top_exp.More():
            current_item = self.top_exp.Current()
            current_item_hash = current_item.__hash__()

            if current_item_hash not in hashes:
                hashes.append(current_item_hash)
                occ_seq.Append(current_item)

            self.top_exp.Next()

        # Convert occ_seq to python list
        occ_iterator = top_tools_list_iterator_of_list_of_shape(occ_seq)
        while occ_iterator.More():
            topology_to_add = self.topology_factory[topology_type](occ_iterator.Value())
            seq.append(topology_to_add)
            occ_iterator.Next()

        if self.ignore_orientation:
            # filter out those entities that share the same TShape, but do *not* share the same orientation
            filter_orientation_seq = []
            for i in seq:
                test_present = False
                for j in filter_orientation_seq:
                    if i.IsSame(j):
                        test_present = True
                        break
                if test_present is False:
                    filter_orientation_seq.append(i)
            return filter_orientation_seq
        return iter(seq)

    def edges(self):
        """
        loops over all edges
        """
        top_abs = import_module("OCC.Core.TopAbs")
        top_abs_edge = getattr(top_abs, "TopAbs_EDGE")
        return self._loop_topology(top_abs_edge)


class SectionInfo:
    """section information"""
    def __init__(self, section, model_id, z_value, z_id, gen_time, x_range):
        self._section_obj = section
        self._model_id = model_id
        self._z_value = z_value
        self._z_id = z_id
        self._gen_time = gen_time
        self._x_range = x_range
        self._edge_num = 0
        self._find_edges()

    def _find_edges(self):
        self._edges = list(Topology(my_shape=self._section_obj).edges())
        self._edge_num = len(self._edges)

    def remove_edges_for_rpc(self):
        self._edges.clear()
        self._edge_num = 0

    def weight(self):
        if self._edge_num == 0:
            raise RuntimeError("All edges have been removed, access prohibited")
        return self._edge_num * self._x_range * self._gen_time

    def has_shape(self):
        return self._edge_num > 0

    def __gt__(self, other):
        return self.weight() > other.weight()

    def __lt__(self, other):
        return self.weight() < other.weight()

    def __eq__(self, other):
        return self.weight() == other.weight

    def section(self):
        return self._section_obj

    def z_value(self):
        return self._z_value

    def z_idx(self):
        return self._z_id

    def edges(self):
        if not self._edges:
            self._find_edges()
        return self._edges

    def model_idx(self):
        return self._model_id


def bbox_for_one_shape(one_shape, tol=1e-6):
    """
    compute bounding box for any TopoDS_*

    Args:
        one_shape (TopoDS_*): any TopoDS_*
        tol (float): tolerance

    Returns:
         list, bounding box
    """
    bnd = import_module("OCC.Core.Bnd")
    bnd_box = getattr(bnd, "Bnd_Box")
    brep_bnd_lib = import_module("OCC.Core.BRepBndLib")
    brep_bnd_lib_add = getattr(brep_bnd_lib, "brepbndlib_Add")

    bbox = bnd_box()
    bbox.SetGap(tol)

    brep_bnd_lib_add(one_shape, bbox)
    x_min, y_min, z_min, x_max, y_max, z_max = bbox.Get()
    return x_min, y_min, z_min, x_max, y_max, z_max


def minimum_distance(shp1, shp2):
    """
    compute minimum distance between 2 BREP's

    Args:
        shp1:    any TopoDS_*
        shp2:    any TopoDS_*

    Returns:
        float, minimum distance
        list, minimum distance points on shp1
        list distance points on shp2
    """
    brep_extrema = import_module("OCC.Core.BRepExtrema")
    brep_extrema_dist_shape_shape = getattr(brep_extrema, "BRepExtrema_DistShapeShape")
    bdss = brep_extrema_dist_shape_shape(shp1, shp2)

    if not bdss.IsDone():
        raise RuntimeError("failed computing minimum distances")

    min_dist = bdss.Value()
    min_dist_shp1, min_dist_shp2 = [], []
    for i in range(1, bdss.NbSolution() + 1):
        min_dist_shp1.append(bdss.PointOnShape1(i))
        min_dist_shp2.append(bdss.PointOnShape2(i))
    return min_dist, min_dist_shp1, min_dist_shp2
