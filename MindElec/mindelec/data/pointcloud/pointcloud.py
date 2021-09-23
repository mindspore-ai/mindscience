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

"""
pointcloud.py supports loading CAD stp format dataset for 3-D physical simulation computing,
and provides operations related to point cloud data.
"""
import os
import re
from enum import IntEnum
from functools import reduce
from importlib import import_module
from math import sqrt, inf
import multiprocessing as mp
from operator import mul
from time import time
import numpy as np
from mindspore import log as logger
from mindelec._c_minddata import PointCloudImpl
from mindelec._c_minddata import PhysicalQuantity as PhyQuantity

from .pointcloud_util import json_2_dict
from .pointcloud_util import load_data
from .pointcloud_util import SectionInfo
from .pointcloud_util import sections_generation
from .pointcloud_util import bbox_for_one_shape
from .pointcloud_util import minimum_distance
from .pointcloud_util import inner_point_generation


class SamplingMode(IntEnum):
    r"""
    Point sampling method, at present support UPPERBOUND(0) and DIMENSIONS(1).

    - 'UPPERBOUND': limit the sampling points number upperbound within whole sampling space, the other space
      parameters such as sampling number on each axis can be automatically computed according to the space
      size ratio.
    - 'DIMENSIONS': users can specify the sampling number in each dimension, the axis order is x:y:z.

    Supported Platforms:
        ``Ascend``
    """
    UPPERBOUND = 0
    DIMENSIONS = 1


class BBoxType(IntEnum):
    r"""
    Bounding box for sampling space, only supports cube-shape sampling space, at present supports STATIC(0) and
    DYNAMIC(1).

    - 'DYNAMIC', generate sampling bbox from the bbox of all 3-D topology models and space extension
      constants, models bbox can be computed automatically after read all files, then add extension
      constants on each direction the DYNAMIC sampling bbox can be obtained. Each model is different.
      Space=(x_min - x_neg, y_min - y_neg, z_min - z_neg, x_max + x_pos, y_max + y_pos, z_max + z_pos)
    - 'STATIC', users can specify the sampling space on each dimension,
      in (x_min, y_min, z_min, x_max, y_max, z_max) order.

    Supported Platforms:
        ``Ascend``
    """
    STATIC = 0
    DYNAMIC = 1


class StdPhysicalQuantity(IntEnum):
    """
    Standard physical quantities fields that Maxwell equations concern about,
    material solving stage will deal with these standard physical fields.

    Supported Platforms:
        ``Ascend``
    """
    MU = 0
    EPSILON = 1
    SIGMA = 2
    TAND = 3


DE_C_INTER_PHYSICAL_QUANTITY = {
    StdPhysicalQuantity.MU: PhyQuantity.DE_PHYSICAL_MU,
    StdPhysicalQuantity.EPSILON: PhyQuantity.DE_PHYSICAL_EPSILON,
    StdPhysicalQuantity.SIGMA: PhyQuantity.DE_PHYSICAL_SIGMA,
    StdPhysicalQuantity.TAND: PhyQuantity.DE_PHYSICAL_TAND,
}


class PointCloudSamplingConfig:
    r"""
    Sampling space config for PointCloud-Tensor generation.

    Args:
        sampling_mode (int): Point sampling method. 0(UPPERBOUND) and 1(DIMENSIONS) are supported.
        bbox_type (int): Bounding box type for sampling space, only supports cube-shape sampling space. 0(STATIC) and
            1(DYNAMIC) are supported.
        mode_args (Union[int, tuple]): sampling upperbound number for SamplingMode. Default: None
        bbox_args (tuple): bounding_box arguments for sampling, has different definition in different bbox_type.
            Default: None

    Raises:
        TypeError: if `sampling_mode` is not an int.
        TypeError: if `bbox_type` is not an int.
        TypeError: if `mode_args` is not one of int or tuple.
        TypeError: if `bbox_args` is not a tuple.
        TypeError:  if `sampling_mode` is 0 but `mode_args` is not int.
        TypeError:  if `sampling_mode` is 1 but `mode_args` is not a tuple of three integers.
        ValueError:  if `sampling_mode` is 1 but the length of `mode_args` is not three.
        ValueError:  if `sampling_mode` not in [0(UPPERBOUND), 1(DIMENSIONS)].
        ValueError:  if `bbox_type` not in [0(STATIC), 1(DYNAMIC)].

    Supported Platforms:
        ``Ascend``
    """
    def __init__(self, sampling_mode, bbox_type, mode_args=None, bbox_args=None):
        if not isinstance(sampling_mode, int) or isinstance(sampling_mode, bool):
            raise TypeError("sampling_mode should be int, but got {} with type {}".format(sampling_mode,
                                                                                          type(sampling_mode)))
        if not isinstance(bbox_type, int) or isinstance(sampling_mode, bool):
            raise TypeError("bbox_type should be int, but got {} with type {}".format(bbox_type, type(bbox_type)))
        if mode_args is not None and not isinstance(mode_args, (int, tuple)) or isinstance(mode_args, bool):
            raise TypeError("mode_args should be int or tuple, but got {} with type {}".format(mode_args,
                                                                                               type(mode_args)))
        if bbox_args is not None and not isinstance(bbox_args, tuple):
            raise TypeError("bbox_args should be tuple, but got {} with type {}".format(bbox_args, type(bbox_args)))
        if sampling_mode not in [0, 1]:
            raise ValueError("Only UPPERBOUND(0) and DIMENSIONS(1) are supported values for sampling_mode, \
                              but got: {} ".format(sampling_mode))
        if bbox_type not in [0, 1]:
            raise ValueError("Only STATIC(0) and DYNAMIC(1) are supported values for bbox_type, \
                              but got: {} ".format(bbox_type))
        if sampling_mode == 0 and (not isinstance(mode_args, int) or isinstance(mode_args, bool)):
            raise TypeError("mode_args should always be int if sampling_mode is \"UPPERBOUND(0)\" , \
                              but got: {} with type {}".format(mode_args, type(mode_args)))
        if sampling_mode == 1:
            _check_mode_args(mode_args)

        self.sampling_mode = sampling_mode
        self.bbox_type = bbox_type
        self.mode_args = mode_args
        self.bbox_args = bbox_args


class MaterialConfig:
    r"""
    Material solution config for PointCloud-Tensor generation, which influence the material solving stage.

    Args:
        json_file (str): Material information for each sub-model json file path.
        material_dir (str): Directory path for all material, physical quantities information of each material
            record in a text file.
        physical_field (dict): Standard physical quantities fields that Maxwell equations concern about,
            material solving stage will deal with these standard physical fields. The key of physical_field dict
            is physical quantity name, the value is default value for this physical quantity.
        customize_physical_field (dict, option): User can specify physical quantities fields according to their
            demand, similarly, material solving stage will take care of them. Default: None
        remove_vacuum (bool, option): Remove sub-solid whose material property is vacuum. Default: True

    Raises:
        TypeError: if `json_file` is not a str.
        TypeError: if `material_dir` is not a str.
        TypeError: if `physical_field` is not a dict.
        TypeError: if `customize_physical_field` is not a dict.
        TypeError: if `remove_vacuum` is not a bool.

    Supported Platforms:
        ``Ascend``
    """
    def __init__(self, json_file, material_dir, physical_field, customize_physical_field=None,
                 remove_vacuum=True):
        if not isinstance(json_file, str):
            raise TypeError("json_file should be str, but got {} with type {}".format(json_file, type(json_file)))
        if not isinstance(material_dir, str):
            raise TypeError("material_dir should be str, but got {} with type {}".format(material_dir,
                                                                                         type(material_dir)))
        if not isinstance(physical_field, dict):
            raise TypeError("physical_field should be dict, but got {} with type {}"
                            .format(physical_field, type(physical_field)))
        if customize_physical_field is not None and not isinstance(customize_physical_field, dict):
            raise TypeError("customize_physical_field should be dict, but got {} with type {}"
                            .format(customize_physical_field, type(customize_physical_field)))
        if not isinstance(remove_vacuum, bool):
            raise TypeError("remove_vacuum should be bool, but got {} with type {}".format(remove_vacuum,
                                                                                           type(remove_vacuum)))
        self.json_file = json_file
        self.material_dir = material_dir
        self.physical_field = dict()
        for k, v in physical_field.items():
            de_key = DE_C_INTER_PHYSICAL_QUANTITY[k]
            self.physical_field[de_key] = v
        self.customize_physical_field = customize_physical_field
        self.remove_vacuum = remove_vacuum

    def print_physical_field(self):
        if self.physical_field is not None:
            for k, v in self.physical_field.items():
                logger.info("key: {0}, value: {1}".format(k, v))
        if self.customize_physical_field is not None:
            for k, v in self.customize_physical_field.items():
                logger.info("key: {0}, value: {1}".format(k, v))


GLOBAL_ALL_SECTIONS = []
GLOBAL_ALL_TENSOR_OUTPUTS = []


def collect_section_result(group_sections):
    global GLOBAL_ALL_SECTIONS
    GLOBAL_ALL_SECTIONS.extend(group_sections)


def collect_space_result(space_solution):
    global GLOBAL_ALL_TENSOR_OUTPUTS
    GLOBAL_ALL_TENSOR_OUTPUTS.extend(space_solution)


def _check_mode_args(inputs):
    """check int types"""
    if not isinstance(inputs, tuple):
        raise TypeError("mode_args should always be tuple if sampling_mode is \"DIMENSIONS(1)\","
                        " but got: {} with type {}".format(inputs, type(inputs)))
    if not len(inputs) == 3:
        raise ValueError("mode_args should always be tuple with length of 3 if sampling_mode is"
                         " \"DIMENSIONS(1)\", but got: {} with length of {}".format(inputs, len(inputs)))
    if not all(isinstance(x, int) and not isinstance(x, bool) for x in inputs):
        raise TypeError("mode_args should always be tuple of 3 integers if sampling_mode is \"DIMENSIONS(1)\","
                        " but got: {} with type ({}, {}, {})".format(inputs, type(inputs[0]),
                                                                     type(inputs[1]), type(inputs[2])))


class PointCloud:
    """
    Read the stp files to generate PointCloud data, for downstream physical-equation AI simulation. Besides, you can
    analyse the space topological information for any 3-D model in stp format. (The most popular format in CAD)

    Args:
        data_dir (str): stp files directory, raw data
        sampling_config (PointCloudSamplingConfig): Sampling space config for PointCloud-Tensor generation.
        material_config (MaterialConfig): Material solution config for PointCloud-Tensor generation, which
            influence the material solving stage.
        num_parallel_workers (int, option): Parallel workers number, this arguments can take effect on all computing
         stages, including reading model, section building, space solving and material solving. Default: os.cpu_count()

    Raises:
        TypeError: if `data_dir` is not a str.
        TypeError: if `sampling_config` is not an instance of class PointCloudSamplingConfig.
        TypeError: if `material_config` is not an instance of class MaterialConfig.
        TypeError: if `num_parallel_workers` is not an int.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindelec.data as md
        >>> from mindelec.data.pointcloud import SamplingMode, BBoxType, StdPhysicalQuantity
        >>> sampling_config = md.PointCloudSamplingConfig(SamplingMode.UPPERBOUND, BBoxType.DYNAMIC, 1000000000,
        ...                                               (5., 5., 5., 5., 5., 5.))
        >>> std_physical_info = {
        ...                         StdPhysicalQuantity.MU: 1.0,
        ...                         StdPhysicalQuantity.EPSILON: 1.0,
        ...                         StdPhysicalQuantity.SIGMA: 0.,
        ...                         StdPhysicalQuantity.TAND: 0.,
        ...                     }
        >>> material_config = md.MaterialConfig(JSON_FILE, MATERIAL_DIR, std_physical_info, None, True)
        >>> pointcloud = md.PointCloud(STP_DIR, sampling_config, material_config)
    """

    def __init__(self, data_dir, sampling_config, material_config, num_parallel_workers=os.cpu_count()):
        if not isinstance(data_dir, str):
            raise TypeError("data_dir should be str, but got {} with type {}".format(data_dir, type(data_dir)))
        if not isinstance(sampling_config, PointCloudSamplingConfig):
            raise TypeError("sampling_config should be an instance of {}, but got {} with type {}"
                            .format(PointCloudSamplingConfig, sampling_config, type(sampling_config)))
        if not isinstance(material_config, MaterialConfig):
            raise TypeError("material_config should be an instance of {}, but got {} with type {}"
                            .format(MaterialConfig, material_config, type(material_config)))
        if not isinstance(num_parallel_workers, int):
            raise TypeError("num_parallel_workers should be int, but got {} with type {}"
                            .format(num_parallel_workers, type(num_parallel_workers)))
        self._data_dir = data_dir
        self._sampling_config = sampling_config
        self._material_config = material_config
        self._num_parallel_workers = num_parallel_workers

        self._whole_model_bbox = None
        self._sub_model_bbox_matrix = None
        self._sampling_distance = list()
        self._sampling_space = dict()
        self._tensor_offset_table = list()

        self._model_list = list()
        self._real_model_index = list()

        self._init_resource()

    def _init_resource(self):
        """
        Private member function, this function is for initialize all computing resources, includes reading raw stp data
        and constructing sampling space
        """
        self._read_step_files()
        self._init_sampling_space()

    def _read_step_files(self):
        """
        Private member function, this function is for reading raw stp data, all these data will be stored in model_list
        member variable.
        """
        logger.info("Start to read stp files.")
        t_read_begin = time()
        if self._data_dir[-1] != '/':
            self._data_dir += '/'
        file_list = [name for name in os.listdir(self._data_dir) if os.path.isfile(self._data_dir + name)
                     and name.endswith(".stp")]

        file_list = self._files_filter(file_list)

        n_file = len(file_list)
        shared_mem_ret = mp.Manager().dict()

        block_size = int(n_file / self._num_parallel_workers)
        block_index = []
        file_groups = [[] for _ in range(self._num_parallel_workers)]

        for i in range(self._num_parallel_workers - 1):
            block_begin = i * block_size
            block_end = (i + 1) * block_size

            file_groups[i] = file_list[block_begin:block_end]
            block_index.append((block_begin, block_end))

        last_begin = (self._num_parallel_workers - 1) * block_size
        file_groups[-1] = file_list[last_begin:]
        block_index.append((last_begin, n_file))

        process_pool0 = mp.Pool(processes=self._num_parallel_workers)

        for group_id, file_group in enumerate(file_groups):
            process_pool0.apply_async(func=load_data, args=(self._data_dir, file_group, group_id, shared_mem_ret))
        process_pool0.close()
        process_pool0.join()

        self._model_list = [None] * n_file
        for key in shared_mem_ret.keys():
            begin = block_index[key][0]
            end = block_index[key][1]
            self._model_list[begin:end] = shared_mem_ret.get(key)
        logger.info("Reading files successfully. Time costs: {}".format(time() - t_read_begin))

    def _files_filter(self, file_list):
        """
        Private member function. This function can find index for all stp file and sort them in numerical order.

        Args:
            file_list (list): Stp file name list in target directory

        Returns:
            list, processed file_list.(After filter and sorter)
        """
        numerical_file_list = []
        vacuum_solid_index = set()
        if self._material_config is not None:
            if self._material_config.remove_vacuum:
                material_dict = json_2_dict(self._material_config.json_file)["solids"]
                for solid in material_dict:
                    if solid["material"] == "Vacuum":
                        vacuum_solid_index.add(solid["index"])
        for file in file_list:
            real_index = re.findall(r"[-+]?\d*\d+|\d+", file)[0]
            if int(real_index) < 0:
                print("[ERROR]Negative index stp file name is prohibited.")
                raise RuntimeError
            if self._material_config is not None and int(real_index) in vacuum_solid_index:
                continue
            numerical_file_list.append((file, int(real_index)))
            self._real_model_index.append(int(real_index))
        numerical_file_list.sort(key=lambda x: x[1])
        self._real_model_index.sort()
        return numerical_file_list

    def _init_sampling_space(self):
        """
        Private member function. This function is for pointcloud sampling space initialization
        """
        real_dimension_number = self._sampling_distance_filter()
        self._space_construct(real_dimension_number)
        logger.info("Sampling space filter has been initialized successfully.")

    def _sampling_distance_filter(self):
        """
        Private member function. Find the sampling number for each axis

        Returns:
            Numpy.array, list of sampling number for each axis in x,y,z order
        """

        self._find_bbox()
        space = []
        if self._sampling_config.bbox_type == BBoxType.DYNAMIC:
            for i in range(3):
                space.append((self._whole_model_bbox[i+3] + self._sampling_config.bbox_args[i+3]) -
                             (self._whole_model_bbox[i] - self._sampling_config.bbox_args[i]))
        elif self._sampling_config.bbox_type == BBoxType.STATIC:
            for i in range(3):
                space.append(self._sampling_config.bbox_args[i+3] - self._sampling_config.bbox_args[i])

        real_sampling_number = []
        if self._sampling_config.sampling_mode == SamplingMode.UPPERBOUND:
            product = reduce(mul, space, 1)
            ratio = pow(self._sampling_config.mode_args / product, 1/3)
            ideal_cube_size = 1 / ratio

            for i in range(3):
                number = max(3, int(space[i] / ideal_cube_size))
                real_sampling_number.append(number)
                self._sampling_distance.append(space[i] / (number - 1))

        elif self._sampling_config.sampling_mode == SamplingMode.DIMENSIONS:
            for i in range(len(self._sampling_config.mode_args)):
                real_sampling_number.append(self._sampling_config.mode_args[i])
                self._sampling_distance.append(space[i] / (real_sampling_number[i] - 1))
        return real_sampling_number

    def _find_bbox(self):
        """
        Private member function, to find the bounding box for original model and all the sub-model
        """
        bbox = []
        bbox_info = np.zeros(shape=(len(self._model_list), 6))
        for model_idx, model in enumerate(self._model_list):
            bbox_info[model_idx] = bbox_for_one_shape(model)
        for column in range(bbox_info.shape[1]):
            if column < 3:
                bbox.append(np.min(bbox_info[:, column], axis=0))
            else:
                bbox.append(np.max(bbox_info[:, column], axis=0))
        self._whole_model_bbox = bbox
        self._sub_model_bbox_matrix = bbox_info

    def _space_construct(self, real_sampling_number):
        """
        Private member function, for construct sampling linear space for pointcloud sampling on each axis

        Args:
            real_sampling_number: Sampling number for each axis in order xyz
        """
        logger.info("Sampling numbers: {}".format(real_sampling_number))
        axis_fields = ['X', 'Y', 'Z']
        if self._sampling_config.bbox_type == BBoxType.DYNAMIC:
            for idx, field in enumerate(axis_fields):
                lower_bound = self._whole_model_bbox[idx] - self._sampling_config.bbox_args[idx]
                upper_bound = self._whole_model_bbox[idx + 3] + self._sampling_config.bbox_args[idx + 3]
                self._sampling_space[field] = np.linspace(start=lower_bound, stop=upper_bound,
                                                          num=real_sampling_number[idx], endpoint=True)
        else:
            for idx, field in enumerate(axis_fields):
                lower_bound = self._sampling_config.bbox_args[idx]
                upper_bound = self._sampling_config.bbox_args[idx + 3]
                self._sampling_space[field] = np.linspace(start=lower_bound, stop=upper_bound,
                                                          num=real_sampling_number[idx], endpoint=True)

    def model_list(self):
        """
        Get model list

        Returns:
            list, model list
        """
        return self._model_list

    def topology_solving(self):
        """
        Solve the topology space by ray-casting algorithm, for each point in sampling space we obtain its sub-model
        belonging, all the results will be stored in a Global list. num_of_workers processes in total will be applied in
        parallel computing.
        """
        logger.info("Start to solve space")
        n_dim = 4
        physical_info_dim = 4   # Init with (x, y, z, model_id), so length = 4, if more material needed, we add it
        if self._material_config is not None:
            physical_info_dim += len(self._material_config.physical_field)
        tensor_shape = [len(self._sampling_space['X']), len(self._sampling_space['Y']),
                        len(self._sampling_space['Z']), physical_info_dim]
        self._offset_base(n_dim, tensor_shape)

        model_groups = self._model_distribution()
        self._section_building(model_groups)

        section_group_vector = self._section_distribution()
        self._space_solving(section_group_vector)

    def tensor_build(self):
        """
        Building pointcloud tensor by using the information obtained in topology_solving module. If poincloud object
        initialized with material config, all material physical info will be considered. All the results will be stored
        in a Global list of dictionary, num_of_workers processes in total will be applied in parallel computing.

        Returns:
            numpy.ndarray, pointcloud result
        """
        physical_info_dim = 4  # Init with (x, y, z, model_id), so length = 4, if more material needed, we add it
        if self._material_config is not None:
            physical_info_dim += len(self._material_config.physical_field)
        tensor_shape = (len(self._sampling_space['X']), len(self._sampling_space['Y']),
                        len(self._sampling_space['Z']), physical_info_dim)

        self._tensor_impl = PointCloudImpl(self._material_config.json_file, self._material_config.material_dir,
                                           self._material_config.physical_field, self._num_parallel_workers)

        tensor = np.zeros(shape=tensor_shape, dtype=np.float64)
        self._tensor_init(tensor)
        self._material_solving(tensor)
        return tensor

    def _offset_base(self, n_dim, shape):
        """
        Compute the pointer offset for the final pointcloud tensor which is numpy.nd_array class

        Args:
            n_dim (int): Dimension number of final nd_array object, it equals to 4
            shape (list): The size on each dimension, in x, y, z, physical_quantities order
        """
        for i in range(n_dim):
            base = 1
            for j in range(i + 1, n_dim, 1):
                base *= shape[j]
            self._tensor_offset_table.append(base)

    def _model_distribution(self):
        """
        Distribution policy for section building parallel compute, after we test several scheduling policies, we find
        out the modular divide policy is optimal in average for most case, actually the topological complexity of each
        sub-model is unpredictable, hence at present we have no more choice to do that.
        We guess maybe it can be refined by pre-collecting the file size for each stp model and estimate the
        corresponding topological complexity, however, it needs more test to prove the effective for this method.

        Returns:
            scheduling result for all stp model, list[list]. This group divide result will be applied in section
            building stage to make the system load as balance as possible.
        """
        total_model_num = len(self._model_list)
        model_group = [[] for _ in range(self._num_parallel_workers)]
        for model_idx in range(total_model_num):
            mod = model_idx % self._num_parallel_workers
            model_group[mod].append((model_idx, self._model_list[model_idx]))
        self._model_list.clear()
        return model_group

    def _section_building(self, distributed_model_groups):
        """
        Private member function, this function is section building module. We use a processes pool API to make use of
        all hardware resource. All sections obtained in this function will be stored in a Global list.

        Important:
            n_block is designed to accelerate the total computing by divide z_sampling space into n_block pieces.
            The idea is for one certain stp model, the complexity is similar among different z_sampling, we will
            introduce this parameter in next function.

        Args:
            distributed_model_groups (list[list]): Scheduled model group generated by self._model_distribution(), each
            process only consider one unique group at a certain time.
        """
        t_section_build_begin = time()
        n_block = 4
        logger.info("Total block number: {}".format(n_block))
        process_pool = mp.Pool(processes=min(os.cpu_count(), n_block * self._num_parallel_workers))

        for model_group_id, model_group in enumerate(distributed_model_groups):
            for block_id in range(n_block):
                process_pool.apply_async(func=sections_generation,
                                         args=(self, model_group, model_group_id, block_id, n_block),
                                         callback=collect_section_result)
        process_pool.close()
        process_pool.join()
        logger.info("Section generation costs: {}.".format(time() - t_section_build_begin))
        logger.info("{} sections have been generated in total.".format(len(GLOBAL_ALL_SECTIONS)))

    def _section_building_impl(self, model_info_list, block_idx, block_number):
        """
        Section building module algorithm implementation. Find the cross section for each stp model on each z-value, we
        use num_of_workers processes to finish this task in parallel mode.

        Args:
            model_info_list (list): A list contains model information.
            block_idx (int): z_sampling block index, we divide z_sampling space into block_number pieces
            block_number (int): Total block number, it equals to n_block as always

        Raises:
            TypeError: if `model_info_list` is not a list.
            TypeError: if `block_idx` is not an int.
            TypeError: if `block_number` is not an int.

        Returns:
            dict[list], A dictionary record the section information obtained by this function. The key is model index,
            the value is a list of section information. section information is a SectionInfo class.
        """
        if not isinstance(model_info_list, list):
            raise TypeError("model_info_list: {} should be list, but got {}"
                            .format(model_info_list, type(model_info_list)))
        if not isinstance(block_idx, int):
            raise TypeError("block_idx: {} should be int, but got {}"
                            .format(block_idx, type(block_idx)))
        if not isinstance(block_number, int):
            raise TypeError("block_number: {} should be int, but got {}"
                            .format(block_number, type(block_number)))
        gp = import_module("OCC.Core.gp")
        gp_dir = getattr(gp, "gp_Dir")
        gp_pln = getattr(gp, "gp_Pln")
        gp_pnt = getattr(gp, "gp_Pnt")

        brep_builder_api = import_module("OCC.Core.BRepBuilderAPI")
        brep_builder_api_make_face = getattr(brep_builder_api, "BRepBuilderAPI_MakeFace")
        brep_algo_api = import_module("OCC.Core.BRepAlgoAPI")
        brep_algo_api_section = getattr(brep_algo_api, "BRepAlgoAPI_Section")

        section_info_dict = dict()
        xy_tolerance = 1.

        size_z = self._sampling_space["Z"].size
        z_sampling = self._sampling_space["Z"][block_idx:size_z:block_number]

        for model_info in model_info_list:
            model_id = model_info[0]
            model = model_info[1]
            section_info_dict[model_id] = list()

            z_min = self._sub_model_bbox_matrix[model_id][2]
            z_max = self._sub_model_bbox_matrix[model_id][5]
            x_range = self._sub_model_bbox_matrix[model_id][3] - self._sub_model_bbox_matrix[model_id][0]
            for z_id, z_value in enumerate(z_sampling):
                if z_value <= z_min:
                    continue
                if z_value >= z_max:
                    break
                t_section_begin = time()

                section_plane = gp_pln(gp_pnt(0, 0, z_value), gp_dir(0, 0, 1))

                section_face = brep_builder_api_make_face(section_plane,
                                                          self._whole_model_bbox[0] - xy_tolerance,
                                                          self._whole_model_bbox[3] + xy_tolerance,
                                                          self._whole_model_bbox[1] - xy_tolerance,
                                                          self._whole_model_bbox[4] + xy_tolerance).Face()

                section = brep_algo_api_section(section_face, model).Shape()
                t_section_end = time()

                real_z_id = block_number * z_id + block_idx
                cur_section_info = SectionInfo(section, model_id, z_value, real_z_id, t_section_end - t_section_begin,
                                               x_range)

                if cur_section_info.has_shape():
                    section_info_dict[model_id].append(cur_section_info)

        return section_info_dict

    def _section_distribution(self):
        """
            Private member function. This function is for generating distribution policy for space solving module
        Distribution policy for space solving parallel compute, after we test several scheduling policies, we find
        out the modular divide policy is the optimal one in average for most case, we use SectionInfo object to express
        the topological complexity for each section, by applying SectionInfo.Weight() method we can obtain that and then
        we divide all sections into different group to make system load more balance hence accelerate computing speed.

        Returns:
            list[list], Scheduled section group, it will be applied in space solving module
        """
        sections_group_vector = [[] for _ in range(self._num_parallel_workers)]

        GLOBAL_ALL_SECTIONS.sort(reverse=True)
        for section_idx, section_info in enumerate(GLOBAL_ALL_SECTIONS):
            group_idx = section_idx % self._num_parallel_workers
            sections_group_vector[group_idx].append(section_info)

        if self._num_parallel_workers != 1:
            for section_info in GLOBAL_ALL_SECTIONS:
                section_info.remove_edges_for_rpc()
        return sections_group_vector

    def _space_solving(self, distributed_section_groups):
        """
        Space solving implementation, we use a processes pool to finish this task in parallel mode.

        Args:
            distributed_section_groups (list[list]): Scheduled section group, each process can deal with one group at
            the same time
        """
        t_space_solving_begin = time()
        process_pool = mp.Pool(processes=min(os.cpu_count(), self._num_parallel_workers))
        for section_info_group in distributed_section_groups:
            process_pool.apply_async(func=inner_point_generation, args=(self, section_info_group),
                                     callback=collect_space_result)
        process_pool.close()
        process_pool.join()
        logger.info("Space solving costs: {}.".format(time() - t_space_solving_begin))
        logger.info("{} line dictionaries have been solved in total.".format(len(GLOBAL_ALL_TENSOR_OUTPUTS)))

    def _inner_point_from_section(self, section_info_obj):
        """
        Generate space solution for one section

        Args:
            section_info_obj(SectionInfo): Topological information for one section

        Raises:
            TypeError: if `section_info_obj` is not an instance of class SectionInfo.

        Returns:
            list[dict], A list of dictionary records the space solution obtained from this module, each element in this
            list is the space solution from one orthogonal line.
        """
        if not isinstance(section_info_obj, SectionInfo):
            raise TypeError("model_info_list: {} should be an instance of {}, but got {}"
                            .format(section_info_obj, SectionInfo, type(section_info_obj)))
        section_result = list()
        tolerance_constant = 1e-6

        z = section_info_obj.z_value()
        idz = section_info_obj.z_idx()
        section_edges = section_info_obj.edges()
        model_idx = section_info_obj.model_idx()

        section_bbox = bbox_for_one_shape(section_info_obj.section(), tolerance_constant)
        left_most = section_bbox[0]
        right_most = section_bbox[3]

        section_edges_x_range = np.zeros(shape=(len(section_edges), 2))

        for edge_id, section_edge in enumerate(section_edges):
            section_edges_x_range[edge_id, 0] = bbox_for_one_shape(section_edge)[0]
            section_edges_x_range[edge_id, 1] = bbox_for_one_shape(section_edge)[3]
        for idx, x in enumerate(self._sampling_space['X']):
            if x <= left_most + sqrt(3) * tolerance_constant:
                continue
            if x >= right_most - sqrt(3) * tolerance_constant:
                break
            one_line_result = self._inner_point_from_line(x, idx, z, idz, model_idx, section_bbox, section_edges,
                                                          section_edges_x_range)
            if one_line_result:
                section_result.append(one_line_result)
        return section_result

    def _get_intersection_coordinate_list(self, idx, x, z, section_edges, edges_x_range, section_bbox):
        """
        get_intersection_coordinate_list
        Args:
            x: The x_value of the line
            idx: The corresponding index of x_value in x_sampling
            z: The z_value of the key section to be solved
            section_bbox: Bounding box of key section
            section_edges: List of edges for key section
            edges_x_range: x_range for each key section edge, hence we got a matrix with shape = [|E|, 2]

        Returns:
            intersection_coordinate_list
        """
        tolerance = 1e-3
        intersection_coordinate_list = []
        down_most = section_bbox[1]
        up_most = section_bbox[4]

        brep_builder_api = import_module("OCC.Core.BRepBuilderAPI")
        brep_builder_api_make_edge = getattr(brep_builder_api, "BRepBuilderAPI_MakeEdge")
        gp = import_module("OCC.Core.gp")
        gp_pnt = getattr(gp, "gp_Pnt")
        orthogonal_edge = brep_builder_api_make_edge(gp_pnt(x, down_most - 1, z),
                                                     gp_pnt(x, up_most + 1, z)).Edge()

        for edge_id, section_edge in enumerate(section_edges):
            if x <= edges_x_range[edge_id, 0] or x >= edges_x_range[edge_id, 1]:
                continue
            min_dist, mp1, _ = minimum_distance(orthogonal_edge, section_edge)
            if min_dist <= tolerance:
                for u in mp1:
                    yu = u.Y()
                    intersection_coordinate_list.append(yu)

        def _judge_yu(inner_yu_list, inner_tolerance, inner_intersection_coordinate_list):
            pre_yu = -inf
            for inner_yu in inner_yu_list:
                delta = inner_yu - pre_yu
                if delta > inner_tolerance:
                    inner_intersection_coordinate_list.append(inner_yu)
                pre_yu = inner_yu
            return inner_intersection_coordinate_list

        if len(intersection_coordinate_list) % 2 == 1 and idx != 0:
            intersection_coordinate_list.clear()
            x_half_left = 0.5 * self._sampling_space['X'][idx - 1] + 0.5 * x
            orthogonal_edge_half_left = brep_builder_api_make_edge(gp_pnt(x_half_left, down_most - 1, z),
                                                                   gp_pnt(x_half_left, up_most + 1, z)).Edge()
            for edge_id, section_edge in enumerate(section_edges):
                if x_half_left <= edges_x_range[edge_id, 0] or x_half_left >= edges_x_range[edge_id, 1]:
                    continue
                min_dist, mp1, _ = minimum_distance(orthogonal_edge_half_left, section_edge)
                if min_dist <= tolerance:
                    if len(mp1) == 1:
                        for u in mp1:
                            yu = u.Y()
                            intersection_coordinate_list.append(yu)
                    else:
                        yu_list = list()
                        for u in mp1:
                            yu_list.append(u.Y())
                        yu_list.sort()
                        intersection_coordinate_list = _judge_yu(yu_list, tolerance, intersection_coordinate_list)

        return intersection_coordinate_list

    def _inner_point_from_line(self, x, idx, z, idz, model_idx, section_bbox, section_edges, edges_x_range):
        """
        Analyse one orthogonal line for a specific section to find out the space solution. The definition of space
        solution is we can determine for all points which sub model which belongs to.

        Args:
            notation:
                we define key section as the current section to be solved
            x: The x_value of the line
            idx: The corresponding index of x_value in x_sampling
            z: The z_value of the key section to be solved
            idz: The corresponding index of z_value in z_sampling
            model_idx: Index of sub-model key section belongs to
            section_bbox: Bounding box of key section
            section_edges: List of edges for key section
            edges_x_range: x_range for each key section edge, hence we got a matrix with shape = [|E|, 2]

        Returns:
            result_dict: A dictionary which can be applied to record the result in following form.
                 key: (idx, idy, idz) ---> offset_value, the value of one point determine the location in final tensor.
                 value: sub-model index determines which key point belongs to
        """
        line_result = dict()
        tolerance = 1e-3

        intersection_coordinate_list = self._get_intersection_coordinate_list(idx, x, z, section_edges, edges_x_range,
                                                                              section_bbox)
        intersection_coordinate_list.sort()
        if len(intersection_coordinate_list) % 2 == 1:
            if len(intersection_coordinate_list) == 1:
                return line_result
            intersection_coordinate_list.pop(1)

        if not intersection_coordinate_list:
            return line_result
        for even in range(0, len(intersection_coordinate_list), 2):
            lower_bound_y = intersection_coordinate_list[even]
            upper_bound_y = intersection_coordinate_list[even + 1]
            for idy, y in enumerate(self._sampling_space['Y']):
                if y <= lower_bound_y + sqrt(3) * tolerance:
                    continue
                if y >= upper_bound_y - sqrt(3) * tolerance:
                    break
                index_p = (idx, idy, idz)
                offset_value = self._offset_compute(index_p)
                line_result[offset_value] = model_idx
        return line_result

    def _offset_compute(self, point_index):
        """
        Private member function, this function is for offset value computing
        Args:
            point_index: (idx, idy, idz) the index in sampling space for one certain point

        Returns:
            int, the offset value computed from point index: (idx, idy, idz)
        """
        offset_value = 0
        for i in range(len(point_index)):
            offset_value += point_index[i] * self._tensor_offset_table[i]
        return offset_value

    def _tensor_init(self, tensor):
        """
        Private member function, this function is for initializing the final tensor.
        Args:
            tensor (ndarray): Final tensor records the pointcloud information
        """
        logger.info("Space initialize begin")
        x_list = list(self._sampling_space['X'])
        y_list = list(self._sampling_space['Y'])
        z_list = list(self._sampling_space['Z'])
        self._tensor_impl.tensor_init_impl(x_list, y_list, z_list, self._tensor_offset_table, tensor)

    def _material_solving(self, tensor):
        """
        Private member function, this function is for assigning material information into final tensor.
        Args:
            tensor (ndarray): Final tensor records the pointcloud information
        Important:
            If construct pointcloud object without material config, this function will raise error in cc layer.
        """
        logger.info("Material analyse begin")
        self._tensor_impl.material_analyse_impl(GLOBAL_ALL_TENSOR_OUTPUTS, self._real_model_index, tensor)
