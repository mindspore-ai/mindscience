# Copyright 2024 Huawei Technologies Co., Ltd
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
# ============================================================================
"""
data
"""
import os
import stat
import time
import pickle
import tqdm
from pymatgen.core.structure import Structure
from pathos.multiprocessing import ProcessingPool as Pool
import numpy as np
from .graph import get_graph, load_orbital_types
from .utils import process_targets


class AijData():
    """
    AijData
    """

    def __init__(self,
                 raw_data_dir: str,
                 graph_dir: str,
                 target: str,
                 dataset_name: str,
                 multiprocessing: bool,
                 radius: float,
                 max_num_nbr: int,
                 edge_aij: bool,
                 default_dtype_np,
                 nums: int = None,
                 inference: bool = False,
                 only_ij: bool = False,
                 load_graph=True):
        """
        :param raw_data_dir: 原始数据目录, 允许存在嵌套
when interface == 'h5',
raw_data_dir
├── 00
│     ├──<target>s.h5
│     ├──element.dat
│     ├──orbital_types.dat
│     ├──site_positions.dat
│     ├──lat.dat
│     └──info.json
├── 01
│     ├──<target>s.h5
│     ├──element.dat
│     ├──orbital_types.dat
│     ├──site_positions.dat
│     ├──lat.dat
│     └──info.json
├── 02
│     ├──<target>s.h5
│     ├──element.dat
│     ├──orbital_types.dat
│     ├──site_positions.dat
│     ├──lat.dat
│     └──info.json
├── ...
        :param graph_dir: 存储图的目录
        :param multiprocessing: 多进程生成图
        :param radius: 生成图的截止半径
        :param max_num_nbr: 生成图限制最大近邻数, 为 0 时不限制
        :param edge_aij: 图的边是否一一对应 Aij, 如果是为 False 则存在两套图的连接, 一套用于节点更新, 一套用于记录 Aij
        :param default_dtype_np: 浮点数数据类型
        """
        self.raw_data_dir = raw_data_dir
        create_from_dft = radius < 0
        radius_info = 'rFromDFT' if create_from_dft else f'{radius}r{max_num_nbr}mn'
        if target == 'hamiltonian':
            graph_file_name = f'HGraph-{dataset_name}-{radius_info}-edge{"" if edge_aij else "!"}=Aij{"-undrct" if only_ij else ""}.pkl'  # undrct = undirected
        elif target == 'density_matrix':
            graph_file_name = f'DMGraph-{dataset_name}-{radius_info}-{edge_aij}edge{"-undrct" if only_ij else ""}.pkl'
        else:
            raise ValueError('Unknown prediction target: {}'.format(target))
        self.data_file = os.path.join(graph_dir, graph_file_name)
        os.makedirs(graph_dir, exist_ok=True)
        self.data, self.slices = None, None
        self.target = target
        self.target_file_name = 'overlaps.h5' if inference else f'{self.target}s.h5'
        self.dataset_name = dataset_name
        self.multiprocessing = multiprocessing
        self.radius = radius
        self.max_num_nbr = max_num_nbr
        self.create_from_dft = create_from_dft
        self.edge_aij = edge_aij
        self.default_dtype_np = default_dtype_np

        self.nums = nums
        self.inference = inference
        self.only_ij = only_ij
        self.transform = None
        self.__indices__ = None
        self.__data_list__ = None
        self._indices = None
        self._data_list = None

        self.data_numpy = None
        self.info = None
        self.max_edge_length = None
        self.min_edge_length = None

        if not os.path.exists(self.data_file):
            self.process()

        if load_graph:
            begin = time.time()

            with open(self.data_file, 'rb') as f:
                loaded_dataset = pickle.load(f)
            self.data_numpy = loaded_dataset[0]
            self.info = loaded_dataset[1]

            print(
                f'Finish loading the processed {len(self.data_numpy)} structures (spinful: {self.info["spinful"]}, '
                f'the number of atomic types: {len(self.info["index_to_Z"])}), cost {time.time() - begin:.2f} seconds')

    @staticmethod
    def element_statistics(data_list):
        """
        calculate element statistics
        """
        index_to_z, _ = np.unique(data_list[0][0], return_inverse=True)
        z_to_index = np.full((100,), -1, dtype=np.int64)

        z_to_index[index_to_z] = np.arange(len(index_to_z))

        length = len(data_list)
        for i in range(length):
            data_list[i][0] = z_to_index[data_list[i][0]].copy()

        return index_to_z, z_to_index, data_list

    @staticmethod
    def mask_padding(data_list_mask):
        """
        mask_padding process
        """
        pad_type = 'constant'
        length = len(data_list_mask)
        for i in range(length):
            padlength = 5314 - data_list_mask[i][1].shape[1]
            data_list_mask[i].append([data_list_mask[i][1].shape[1]])
            data_list_mask[i][1] = np.pad(data_list_mask[i][1], ((0, 0), (0, padlength)), pad_type)
            data_list_mask[i][2] = np.pad(data_list_mask[i][2], ((0, padlength), (0, 0)), pad_type)
            data_list_mask[i][6] = np.pad(data_list_mask[i][6], ((0, padlength), (0, 0)), pad_type)
            data_list_mask[i][9] = np.pad(data_list_mask[i][9], ((0, padlength), (0, 0)), pad_type)
            data_list_mask[i][10] = np.pad(data_list_mask[i][10], ((0, padlength), (0, 0)),
                                           pad_type,
                                           constant_values=False)
            data_list_mask[i][3] = np.array([0])
            mask_dim1 = np.append(np.ones(data_list_mask[i][11][0], dtype=np.float32),
                                  np.zeros(5314 - data_list_mask[i][11][0], dtype=np.float32))
            mask_dim2 = mask_dim1.reshape((5314, 1))
            mask_dim3 = mask_dim1.reshape((5314, 1, 1))
            data_list_mask[i].append(mask_dim1)
            data_list_mask[i].append(mask_dim2)
            data_list_mask[i].append(mask_dim3)
        return data_list_mask

    def process(self):
        """
        sub process of process worker
        """
        begin = time.time()
        folder_list = []
        for root, _, files in os.walk(self.raw_data_dir):
            if {'element.dat', 'orbital_types.dat', 'lat.dat', 'site_positions.dat'}.issubset(files):
                if self.target_file_name in files:
                    folder_list.append(root)
        folder_list = folder_list[:self.nums]
        if not folder_list:
            raise AssertionError("Can not find any structure")
        begin = time.time()

        self.multiprocessing = True
        if self.multiprocessing:
            with Pool() as pool:
                data_list = list(tqdm.tqdm(pool.imap(self.process_worker, folder_list), total=len(folder_list)))
        else:
            data_list = [self.process_worker(folder) for folder in tqdm.tqdm(folder_list)]

        print('Finish processing %d structures, have cost %d seconds' % (len(data_list), time.time() - begin))

        index_to_z, z_to_index, data_list = self.element_statistics(data_list)

        spinful = data_list[0][-1]

        _, orbital_types = load_orbital_types(path=os.path.join(folder_list[0], 'orbital_types.dat'),
                                              return_orbital_types=True)

        elements = np.loadtxt(os.path.join(folder_list[0], 'element.dat'))
        orbital_types_new = []
        length = len(index_to_z)
        for i in range(length):
            orbital_types_new.append(orbital_types[np.where(elements == index_to_z[i])[0][0]])

        zip_data = [
            data_list,
            dict(spinful=spinful, index_to_Z=index_to_z, Z_to_index=z_to_index, orbital_types=orbital_types_new)
        ]

        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        modes = stat.S_IWUSR | stat.S_IRUSR
        with os.fdopen(os.open(self.data_file, flags, modes), 'wb') as f:
            pickle.dump(zip_data, f)

        print('Finished saving %d structures to save_graph_dir, have cost %d seconds' %
              (len(data_list), time.time() - begin))

    def process_worker(self, folder):
        """
        process worker to load the data
        """
        stru_id = os.path.split(folder)[-1]

        site_positions = np.loadtxt(os.path.join(folder, 'site_positions.dat')).T
        elements = np.loadtxt(os.path.join(folder, 'element.dat'))
        if elements.ndim == 0:
            elements = elements[None]
            site_positions = site_positions[None, :]
        structure = Structure(np.loadtxt(os.path.join(folder, 'lat.dat')).T,
                              elements,
                              site_positions,
                              coords_are_cartesian=True,
                              to_unit_cell=False)

        cart_coords = structure.cart_coords
        numbers = np.array(structure.atomic_numbers)
        structure.lattice.matrix.setflags(write=True)
        lattice = structure.lattice.matrix

        return get_graph(cart_coords,
                         numbers,
                         stru_id,
                         edge_aij=self.edge_aij,
                         lattice=lattice,
                         default_dtype_np=self.default_dtype_np,
                         data_folder=folder,
                         target_file_name=self.target_file_name,
                         inference=self.inference,
                         only_ij=self.only_ij,
                         create_from_dft=self.create_from_dft)

    def set_mask(self, targets, del_aij=True):
        """
        set mask process
        """
        dtype = np.float32
        equivariant_blocks, out_js_list, out_slices = process_targets(self.info['orbital_types'],
                                                                      self.info["index_to_Z"], targets)
        index_x = "x"
        index_edge_index = "edge_index"
        index_edge_attr = "edge_attr"
        index_stru_id = "stru_id"
        index_pos = "pos"
        index_lattice = "lattice"
        index_edge_key = "edge_key"
        index_aij = "Aij"
        index_aij_mask = "Aij_mask"
        index_atom_num_orbital = "atom_num_orbital"
        index_spinful = "spinful"

        data_list_mask = []
        index = 0
        min_edge_length = 100000
        max_edge_length = -1
        for data in self.data_numpy:
            min_edge_length = min(min_edge_length, len(data[1][0]))
            max_edge_length = max(max_edge_length, len(data[1][0]))
            index += 1
            map_dict = {
                index_x: 0,
                index_edge_index: 1,
                index_edge_attr: 2,
                index_stru_id: 3,
                index_pos: 4,
                index_lattice: 5,
                index_edge_key: 6,
                index_aij: 7,
                index_aij_mask: 8,
                index_atom_num_orbital: 9,
                index_spinful: 10
            }

            num_edges = len(data[map_dict[index_edge_attr]])
            # label of each edge is a vector which is each target H block flattened and concatenated

            label = np.zeros((num_edges, out_slices[-1]), dtype=dtype)
            mask = np.zeros((num_edges, out_slices[-1]), dtype=np.int8)

            atomic_number_edge_i = data[map_dict.get('x', None)][data[map_dict.get(index_edge_index, None)][0]]
            atomic_number_edge_j = data[map_dict.get('x', None)][data[map_dict.get(index_edge_index, None)][1]]

            for index_out, equivariant_block in enumerate(equivariant_blocks):
                for n_m_str, block_slice in equivariant_block.items():
                    condition_atomic_number_i, condition_atomic_number_j = map(
                        lambda x: self.info["Z_to_index"][int(x)], n_m_str.split())
                    condition_slice_i = slice(block_slice[0], block_slice[1])
                    condition_slice_j = slice(block_slice[2], block_slice[3])
                    if data[map_dict.get(index_aij, None)] is not None:
                        out_slice = slice(out_slices[index_out], out_slices[index_out + 1])
                        condition_index = np.where((atomic_number_edge_i == condition_atomic_number_i)
                                                   & (atomic_number_edge_j == condition_atomic_number_j))

                        label[condition_index[0], out_slice] += \
                            data[map_dict.get(index_aij, None)][:, condition_slice_i, condition_slice_j].reshape(
                                num_edges, -1)[condition_index]
                        mask[condition_index[0], out_slice] += 1

            if del_aij:
                data.pop(map_dict.get(index_aij_mask, None))

            map_dict = {
                index_x: 0,
                index_edge_index: 1,
                index_edge_attr: 2,
                index_stru_id: 3,
                index_pos: 4,
                index_lattice: 5,
                index_edge_key: 6,
                index_aij: 7,
                index_atom_num_orbital: 8,
                index_spinful: 9
            }

            if data[map_dict.get(index_aij, None)] is not None:
                data.append(label)
                mask = mask.astype(dtype=bool)
                data.append(mask)
                if del_aij:
                    data.pop(map_dict.get(index_aij, None))
            data_list_mask.append(data)

        self.max_edge_length = max_edge_length

        self.data_numpy = self.mask_padding(data_list_mask)
        return out_js_list, out_slices
