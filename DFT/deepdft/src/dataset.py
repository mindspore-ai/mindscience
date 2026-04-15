# Copyright 2021 Huawei Technologies Co., Ltd
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""dataset"""
from typing import List, Optional
import gzip
import tarfile
import tempfile
import multiprocessing
import queue
import time
import threading
import logging
import zlib
import os
import io
import math
import lz4.frame
import numpy as np
import ase
import ase.neighborlist
import ase.io.cube
import ase.units
from ase.calculators.vasp import VaspChargeDensity
import asap3
import mindspore as ms
from mindspore import Tensor

from .layer import pad_and_stack, pad_and_stack_np


def _cell_heights(cell_object):
    volume = cell_object.volume
    crossproducts = np.cross(cell_object[[1, 2, 0]], cell_object[[2, 0, 1]])
    crosslengths = np.sqrt(np.sum(np.square(crossproducts), axis=1))
    heights = volume / crosslengths
    return heights


def rotating_pool_worker(dataset, rng, queue):
    while True:
        for index in rng.permutation(len(dataset)).tolist():
            queue.put(dataset[index])


def transfer_thread(queue: multiprocessing.Queue, datalist: list):
    while True:
        for index in range(len(datalist)):
            datalist[index] = queue.get()


class RotatingPoolData:
    """Wrapper for a dataset that continously loads data into a smaller pool.
    The data loading is performed in a separate process and is assumed to be IO bound."""

    def __init__(self, dataset, pool_size):
        super(RotatingPoolData).__init__()
        self.pool_size = pool_size
        self.parent_data = dataset
        self.rng = np.random.default_rng()
        logging.debug("Filling rotating data pool of size %d" % pool_size)
        self.data_pool = [
            self.parent_data[i]
            for i in self.rng.integers(
                0, high=len(self.parent_data), size=self.pool_size, endpoint=False
            ).tolist()
        ]
        self.loader_queue = multiprocessing.Queue(2)

        # Start loaders
        self.loader_process = multiprocessing.Process(
            target=rotating_pool_worker,
            args=(self.parent_data, self.rng, self.loader_queue),
            daemon=True,
        )
        self.transfer_thread = threading.Thread(
            target=transfer_thread, args=(self.loader_queue, self.data_pool), daemon=True
        )
        self.loader_process.start()
        self.transfer_thread.start()

    def __len__(self):
        return self.pool_size

    def __getitem__(self, index):
        return self.data_pool[index]


class BufferData:
    """Wrapper for a dataset. Loads all data into memory."""

    def __init__(self, dataset, args, set_pbc, num_probes):
        self.convert = CollateFuncRandomSample(args.cutoff, num_probes, set_pbc)
        self.data_objects = [self.convert(dataset[i]) for i in range(len(dataset))]

    def __len__(self):
        return len(self.data_objects)

    def __getitem__(self, index):
        return self.data_objects[index]


# AttributeError: 'DensityData' object has no attribute 'parent'
class DensityData:
    def __init__(self, datapath=None, data=None):
        if data is not None:
            self.data = data
        elif datapath is not None:
            if os.path.isfile(datapath) and datapath.endswith(".tar"):
                self.data = DensityDataTar(datapath)
            elif os.path.isdir(datapath):
                self.data = DensityDataDir(datapath)
            else:
                raise ValueError("Did not find dataset at path %s", datapath)
        else:
            raise ValueError("Must provide either datapath or data")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def take(self, indices):
        if not isinstance(indices, (list, tuple)):
            raise ValueError("indices must be an iterable (e.g., list or tuple)")
        return DensityData(data=self.data.take(indices))
    
    def concat(self, other):
        self.data.concat(other.data)


class DensityDataDir:
    def __init__(self, directory=None, member_list=None):
        if member_list is not None:
            self.member_list = member_list
        else:
            self.directory = directory
            self.member_list = [(dir, mem) for mem in sorted(os.listdir(self.directory))]
        self.key_to_idx = {str(k[1]): i for i, k in enumerate(self.member_list)}

    def __len__(self):
        return len(self.member_list)

    def extractfile(self, member):
        directory, filename = member
        path = os.path.join(directory, filename)

        filecontent = _decompress_file(path)
        if path.endswith((".cube", ".cube.gz", ".cube.zz", "cube.lz4")):
            density, atoms, origin = _read_cube(filecontent)
        else:
            density, atoms, origin = _read_vasp(filecontent)

        grid_pos = _calculate_grid_pos(density, origin, atoms.get_cell())

        metadata = {"filename": filename}
        return {
            "density": density,
            "atoms": atoms,
            "origin": origin,
            "grid_position": grid_pos,
            "metadata": metadata,  # Meta information
        }

    def __getitem__(self, index):
        if isinstance(index, str):
            index = self.key_to_idx[index]
        return self.extractfile(self.member_list[index])
    
    def take(self, indices):
        if not isinstance(indices, (list, tuple)):
            raise ValueError("indices must be an iterable (e.g., list or tuple)")
        return DensityDataDir(member_list=[self.member_list[i] for i in indices])

    def concat(self, other):
        self.member_list += other.member_list


class DensityDataTar:
    def __init__(self, tarpath=None, member_list=None):
        if member_list is not None:
            self.member_list = member_list
        else:
            self.tarpath = tarpath
            self.member_list = []
            # Index tar file
            with tarfile.open(self.tarpath, "r:") as tar:
                for member in tar.getmembers():
                    self.member_list.append([tarpath, member])

        self.key_to_idx = {str(k[1]): i for i, k in enumerate(self.member_list)}

    def __len__(self):
        return len(self.member_list)

    def extract_member(self, member):  # trainfo="<TarInfo '003999.CHGCAR.lz4' at 0x7fa3160cddc0>"
        tarpath, tarinfo = member
        with tarfile.open(tarpath, "r") as tar:
            filecontent = _decompress_tarmember(tar, tarinfo)
            if tarinfo.name.endswith((".cube", ".cube.gz", "cube.zz", "cube.lz4")):
                density, atoms, origin = _read_cube(filecontent)
            else:
                density, atoms, origin = _read_vasp(filecontent)

        grid_pos = _calculate_grid_pos(density, origin, atoms.get_cell())

        metadata = {"filename": tarinfo.name}
        return {
            "density": density,         # numpy.ndarray float64
            "atoms": atoms,             # Atoms class e.g. Atoms(symbols='C4N2OH8', pbc=True, cell=[6.971137, 9.870693, 7.574937])
            "origin": origin,           # numpy.ndarray float64
            "grid_position": grid_pos,  # numpy.ndarray float64
            "metadata": metadata,       # Meta information
        }

    def __getitem__(self, index):
        if isinstance(index, str):
            index = self.key_to_idx[index]
        return self.extract_member(self.member_list[index])
    
    def take(self, indices):
        if not isinstance(indices, (list, tuple)):
            raise ValueError("indices must be an iterable (e.g., list or tuple)")
        return DensityDataTar(member_list=[self.member_list[i] for i in indices])
    
    def concat(self, other):
        self.member_list += other.member_list


class AseNeigborListWrapper:
    """Wrapper around ASE neighborlist to have the same interface as asap3 neighborlist"""

    def __init__(self, cutoff, atoms):
        self.neighborlist = ase.neighborlist.NewPrimitiveNeighborList(
            cutoff, skin=0.0, self_interaction=False, bothways=True
        )
        self.neighborlist.build(
            atoms.get_pbc(), atoms.get_cell(), atoms.get_positions()
        )
        self.cutoff = cutoff
        self.atoms_positions = atoms.get_positions()
        self.atoms_cell = atoms.get_cell()

    def get_neighbors(self, i, cutoff):
        assert (
                cutoff == self.cutoff
        ), "Cutoff must be the same as used to initialise the neighborlist"

        indices, offsets = self.neighborlist.get_neighbors(i)

        rel_positions = (
                self.atoms_positions[indices]
                + offsets @ self.atoms_cell
                - self.atoms_positions[i][None]
        )

        dist2 = np.sum(np.square(rel_positions), axis=1)

        return indices, rel_positions, dist2


def grid_iterator_worker(atoms, meshgrid, probe_count, cutoff, slice_id_queue, result_queue):
    try:
        neighborlist = asap3.FullNeighborList(cutoff, atoms)
    except Exception as e:
        logging.info("Failed to create asap3 neighborlist, this might be very slow. Error: %s", e)
        neighborlist = None
    while True:
        try:
            slice_id = slice_id_queue.get(True, 1)
        except queue.Empty:
            while not result_queue.empty():
                time.sleep(1)
            result_queue.close()
            return 0
        res = DensityGridIterator.static_get_slice(slice_id, atoms, meshgrid, probe_count, cutoff,
                                                   neighborlist=neighborlist)
        result_queue.put((slice_id, res))


class DensityGridIterator:
    def __init__(self, densitydict, probe_count: int, cutoff: float, set_pbc_to: Optional[bool] = None):
        num_positions = np.prod(densitydict["grid_position"].shape[0:3])
        self.num_slices = int(math.ceil(num_positions / probe_count))
        self.probe_count = probe_count
        self.cutoff = cutoff
        self.set_pbc = set_pbc_to

        if self.set_pbc is not None:
            self.atoms = densitydict["atoms"].copy()
            self.atoms.set_pbc(self.set_pbc)
        else:
            self.atoms = densitydict["atoms"]

        self.meshgrid = densitydict["grid_position"]

    def get_slice(self, slice_index):
        return self.static_get_slice(slice_index, self.atoms, self.meshgrid, self.probe_count, self.cutoff)

    @staticmethod
    def static_get_slice(slice_index, atoms, meshgrid, probe_count, cutoff, neighborlist=None):
        num_positions = np.prod(meshgrid.shape[0:3])
        flat_index = np.arange(slice_index * probe_count, min((slice_index + 1) * probe_count, num_positions))
        pos_index = np.unravel_index(flat_index, meshgrid.shape[0:3])
        probe_pos = meshgrid[pos_index]
        probe_edges, probe_edges_displacement = probes_to_graph(atoms, probe_pos, cutoff, neighborlist)

        if not probe_edges:
            probe_edges = [np.zeros((0, 2), dtype=np.int32)]
            probe_edges_displacement = [np.zeros((0, 3), dtype=np.float32)]

        res = {
            "probe_edges"               :  ms.Tensor(np.concatenate(probe_edges, axis=0).astype(np.int32), dtype=ms.int32),
            "probe_edges_displacement"  :  ms.Tensor(np.concatenate(probe_edges_displacement, axis=0).astype(np.float32), dtype=ms.float32),
        }
        res["num_probe_edges"] = ms.Tensor(np.array(res["probe_edges"].shape[0], dtype=np.int32), dtype=ms.int32)
        res["num_probes"]      = ms.Tensor(np.array(len(flat_index), dtype=np.int32), dtype=ms.int32)
        res["probe_xyz"]       = ms.Tensor(probe_pos.astype(np.float32), dtype=ms.float32)
        return res

    def __iter__(self):
        self.current_slice = 0
        slice_id_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue(100)
        self.finished_slices = dict()
        for i in range(self.num_slices):
            slice_id_queue.put(i)
        self.workers = [multiprocessing.Process(
            target=grid_iterator_worker,
            args=(self.atoms, self.meshgrid, self.probe_count, self.cutoff, slice_id_queue, self.result_queue)) for _ in
            range(6)
        ]
        for w in self.workers:
            w.start()
        return self

    def __next__(self):
        if self.current_slice < self.num_slices:
            this_slice = self.current_slice
            self.current_slice += 1

            # Retrieve finished slices until we get the one we are looking for
            while this_slice not in self.finished_slices:
                i, res = self.result_queue.get()
                dic = {}
                for k, v in res.items():
                    if v.shape and v.shape[0] == 0:
                        dic[k] = ms.ops.zeros(v.shape)
                    else:
                        dic[k] = ms.Tensor(v)
                res = dic

                self.finished_slices[i] = res
            return self.finished_slices.pop(this_slice)
        else:
            for w in self.workers:
                w.join()
            raise StopIteration


def atoms_and_probe_sample_to_graph_dict(density, atoms, grid_pos, cutoff, num_probes):
    # Sample probes on the calculated grid
    probe_choice_max = np.prod(grid_pos.shape[0:3])
    probe_choice = np.random.randint(probe_choice_max, size=num_probes)
    probe_choice = np.unravel_index(probe_choice, grid_pos.shape[0:3])
    probe_pos = grid_pos[probe_choice]
    probe_target = density[probe_choice]

    atom_edges, atom_edges_displacement, neighborlist, inv_cell_T = atoms_to_graph(atoms, cutoff)
    probe_edges, probe_edges_displacement = probes_to_graph(atoms, probe_pos, cutoff, neighborlist=neighborlist,
                                                            inv_cell_T=inv_cell_T)

    if not probe_edges:
        probe_edges = [np.zeros((0, 2), dtype=np.int32)] 
        probe_edges_displacement = [np.zeros((0, 3), dtype=np.int32)]
    # pylint: disable=E1102
    default_type = np.float32
    res = {
        "nodes": np.array(atoms.get_atomic_numbers(), dtype=np.int32),
        "atom_edges": np.concatenate(atom_edges, axis=0).astype(default_type),
        "atom_edges_displacement": np.concatenate(atom_edges_displacement, axis=0).astype(default_type),
        "probe_edges": np.concatenate(probe_edges, axis=0),
        "probe_edges_displacement": np.concatenate(probe_edges_displacement, axis=0).astype(default_type),
        "probe_target": probe_target.astype(default_type),
    }
    res["num_nodes"] = np.array(res["nodes"].shape[0], dtype=np.int32)
    res["num_atom_edges"] = np.array(res["atom_edges"].shape[0], dtype=np.int32)
    res["num_probe_edges"] = np.array(res["probe_edges"].shape[0], dtype=np.int32)
    res["num_probes"] = np.array(res["probe_target"].shape[0], dtype=np.int32)
    res["probe_xyz"] = probe_pos.astype(default_type)
    res["atom_xyz"] = atoms.get_positions().astype(default_type)
    res["cell"] = np.array(atoms.get_cell())

    return res


def atoms_to_graph_dict(atoms, cutoff):
    atom_edges, atom_edges_displacement, _, _ = atoms_to_graph(atoms, cutoff)    
    # pylint: disable=E1102
    default_type = np.float32
    res = {
        "nodes": np.array(atoms.get_atomic_numbers(), dtype=np.int32),
        "atom_edges": np.concatenate(atom_edges, axis=0).astype(default_type),
        "atom_edges_displacement": np.concatenate(atom_edges_displacement, axis=0).astype(default_type),
    }
    res["num_nodes"] = np.array(res["nodes"].shape[0], dtype=np.int32)
    res["num_atom_edges"] = np.array(res["atom_edges"].shape[0], dtype=np.int32)
    res["atom_xyz"] = atoms.get_positions().astype(default_type)
    res["cell"] = np.array(atoms.get_cell())

    return res


def atoms_to_graph(atoms, cutoff):
    atom_edges = []
    atom_edges_displacement = []

    inv_cell_T = np.linalg.inv(atoms.get_cell().complete().T)

    # Compute neighborlist
    if (
            np.any(atoms.get_cell().lengths() <= 0.0001)
            or (
            np.any(atoms.get_pbc())
            and np.any(_cell_heights(atoms.get_cell()) < cutoff)
    )
    ):
        neighborlist = AseNeigborListWrapper(cutoff, atoms)
    else:
        neighborlist = asap3.FullNeighborList(cutoff, atoms)

    atom_positions = atoms.get_positions()

    for i in range(len(atoms)):
        neigh_idx, neigh_vec, _ = neighborlist.get_neighbors(i, cutoff)

        self_index = np.ones_like(neigh_idx) * i
        edges = np.stack((neigh_idx, self_index), axis=1)

        neigh_pos = atom_positions[neigh_idx]
        this_pos = atom_positions[i]
        neigh_origin = neigh_vec + this_pos - neigh_pos
        neigh_origin_scaled = np.round(inv_cell_T.dot(neigh_origin.T).T)

        atom_edges.append(edges)
        atom_edges_displacement.append(neigh_origin_scaled)

    return atom_edges, atom_edges_displacement, neighborlist, inv_cell_T


def probes_to_graph(atoms, probe_pos, cutoff, neighborlist=None, inv_cell_T=None):
    probe_edges = []
    probe_edges_displacement = []
    if inv_cell_T is None:
        inv_cell_T = np.linalg.inv(atoms.get_cell().complete().T)

    if hasattr(neighborlist, "get_neighbors_querypoint"):
        results = neighborlist.get_neighbors_querypoint(probe_pos, cutoff)
        atomic_numbers = atoms.get_atomic_numbers()
    else:
        # Insert probe atoms
        num_probes = probe_pos.shape[0]
        probe_atoms = ase.Atoms(numbers=[0] * num_probes, positions=probe_pos)
        atoms_with_probes = atoms.copy()
        atoms_with_probes.extend(probe_atoms)
        atomic_numbers = atoms_with_probes.get_atomic_numbers()

        if (
                np.any(atoms.get_cell().lengths() <= 0.0001)
                or (
                np.any(atoms.get_pbc())
                and np.any(_cell_heights(atoms.get_cell()) < cutoff)
        )
        ):
            neighborlist = AseNeigborListWrapper(cutoff, atoms_with_probes)
        else:
            neighborlist = asap3.FullNeighborList(cutoff, atoms_with_probes)

        results = [neighborlist.get_neighbors(i + len(atoms), cutoff) for i in range(num_probes)]

    atom_positions = atoms.get_positions()
    for i, (neigh_idx, neigh_vec, _) in enumerate(results):
        neigh_atomic_species = atomic_numbers[neigh_idx]

        neigh_is_atom = neigh_atomic_species != 0
        neigh_atoms = neigh_idx[neigh_is_atom]
        self_index = np.ones_like(neigh_atoms) * i
        edges = np.stack((neigh_atoms, self_index), axis=1)

        neigh_pos = atom_positions[neigh_atoms]
        this_pos = probe_pos[i]
        neigh_origin = neigh_vec[neigh_is_atom] + this_pos - neigh_pos
        neigh_origin_scaled = np.round(inv_cell_T.dot(neigh_origin.T).T)

        probe_edges.append(edges)
        probe_edges_displacement.append(neigh_origin_scaled)

    return probe_edges, probe_edges_displacement


def preprocess(graphs, has_probe=True):
    default_type = ms.float32
    type_dict = {
        "nodes": ms.int32,
        "atom_edges": ms.int32,
        "atom_edges_displacement": default_type,
        "num_nodes": ms.int32,
        "num_atom_edges": ms.int32,
        "atom_xyz": default_type,
        "cell": default_type,
        "probe_edges": ms.int32,
        "probe_edges_displacement": default_type,
        "probe_target": default_type,
        "num_probe_edges": ms.int32,
        "num_probes": ms.int32,
        "probe_xyz": default_type,
    }
    for k, v in graphs.items():
        if k in type_dict:
            graphs[k] = ms.Tensor(v, dtype=type_dict[k])
    # graphs["nodes"] = ms.Tensor(graphs["nodes"], dtype=ms.int32)
    # graphs["atom_edges"] = ms.Tensor(graphs["atom_edges"], dtype=ms.int32)
    # graphs["atom_edges_displacement"] = ms.Tensor(graphs["atom_edges_displacement"], dtype=default_type)
    # graphs["num_nodes"] = ms.Tensor(graphs["num_nodes"], dtype=ms.int32)
    # graphs["num_atom_edges"] = ms.Tensor(graphs["num_atom_edges"], dtype=ms.int32)
    # graphs["atom_xyz"] = ms.Tensor(graphs["atom_xyz"], dtype=default_type)
    # graphs["cell"] = ms.Tensor(graphs["cell"], dtype=default_type)

    # try:
    #     graphs["probe_edges"] = ms.Tensor(graphs["probe_edges"], dtype=ms.int32)
    #     graphs["probe_edges_displacement"] = ms.Tensor(graphs["probe_edges_displacement"], dtype=default_type)
    #     graphs["probe_target"] = ms.Tensor(graphs["probe_target"], dtype=default_type)
    #     graphs["num_probe_edges"] = ms.Tensor(graphs["num_probe_edges"], dtype=ms.int32)
    #     graphs["num_probes"] = ms.Tensor(graphs["num_probes"], dtype=ms.int32)
    #     graphs["probe_xyz"] = ms.Tensor(graphs["probe_xyz"], dtype=default_type)
    # except:
    #     pass

    return graphs

def collate_list_of_dicts(list_of_dicts, batchinfo):
    # Convert from "list of dicts" to "dict of lists"
    dict_of_lists = {k: [dic[k] for dic in list_of_dicts] for k in list_of_dicts[0]}

    # Convert each list of tensors to single tensor with pad and stack
    collated = {k: pad_and_stack_np(dict_of_lists[k]) for k in dict_of_lists}
    return preprocess(collated)

def collate_list_of_dicts_ms(list_of_dicts):
    # Convert from "list of dicts" to "dict of lists"
    dict_of_lists = {k: [dic[k] for dic in list_of_dicts] for k in list_of_dicts[0]}

    # Convert each list of tensors to single tensor with pad and stack
    collated = {k: pad_and_stack(dict_of_lists[k]) for k in dict_of_lists}
    return preprocess(collated)


class CollateFuncRandomSample:
    def __init__(self, cutoff, num_probes, set_pbc_to=None):
        self.num_probes = num_probes
        self.cutoff = cutoff
        self.set_pbc = set_pbc_to

    def __call__(self, input_dicts: List):
        if not isinstance(input_dicts, list):
            input_dicts = [input_dicts]
        graphs = []
        for i in input_dicts:
            if self.set_pbc is not None:
                atoms = i["atoms"].copy()
                atoms.set_pbc(self.set_pbc)
            else:
                atoms = i["atoms"]

            graphs.append(atoms_and_probe_sample_to_graph_dict(
                i["density"],
                atoms,
                i["grid_position"],
                self.cutoff,
                self.num_probes,
            ))
        del input_dicts
        return graphs


class CollateFuncAtoms:
    def __init__(self, cutoff, set_pbc_to=None):
        self.cutoff = cutoff
        self.set_pbc = set_pbc_to

    def __call__(self, input_dicts: List):
        if not isinstance(input_dicts, list):
            input_dicts = [input_dicts]
        graphs = []
        for i in input_dicts:
            if self.set_pbc is not None:
                atoms = i["atoms"].copy()
                atoms.set_pbc(self.set_pbc)
            else:
                atoms = i["atoms"]

            graphs.append(atoms_to_graph_dict(
                atoms,
                self.cutoff,
            ))
        del input_dicts
        return graphs


def _calculate_grid_pos(density, origin, cell):
    # Calculate grid positions
    ngridpts = np.array(density.shape)  # grid matrix
    grid_pos = np.meshgrid(
        np.arange(ngridpts[0]) / density.shape[0],
        np.arange(ngridpts[1]) / density.shape[1],
        np.arange(ngridpts[2]) / density.shape[2],
        indexing="ij",
    )
    grid_pos = np.stack(grid_pos, 3)
    grid_pos = np.dot(grid_pos, cell)
    grid_pos = grid_pos + origin
    return grid_pos


def _decompress_tarmember(tar, tarinfo):
    """Extract compressed tar file member and return a bytes object with the content"""

    bytesobj = tar.extractfile(tarinfo).read()
    if tarinfo.name.endswith(".zz"):
        filecontent = zlib.decompress(bytesobj)
    elif tarinfo.name.endswith(".lz4"):
        filecontent = lz4.frame.decompress(bytesobj)
    elif tarinfo.name.endswith(".gz"):
        filecontent = gzip.decompress(bytesobj)
    else:
        filecontent = bytesobj

    return filecontent


def _decompress_file(filepath):
    if filepath.endswith(".zz"):
        with open(filepath, "rb") as fp:
            f_bytes = fp.read()
        filecontent = zlib.decompress(f_bytes)
    elif filepath.endswith(".lz4"):
        with lz4.frame.open(filepath, mode="rb") as fp:
            filecontent = fp.read()
    elif filepath.endswith(".gz"):
        with gzip.open(filepath, mode="rb") as fp:
            filecontent = fp.read()
    else:
        with open(filepath, mode="rb") as fp:
            filecontent = fp.read()
    return filecontent


def _read_vasp(filecontent):
    # Write to tmp file and read using ASE
    tmpfd, tmppath = tempfile.mkstemp(prefix="tmpdeepdft")
    tmpfile = os.fdopen(tmpfd, "wb")
    tmpfile.write(filecontent)
    tmpfile.close()
    vasp_charge = VaspChargeDensity(filename=tmppath)
    os.remove(tmppath)
    density = vasp_charge.chg[-1]  # separate density
    atoms = vasp_charge.atoms[-1]  # separate atom positions

    return density, atoms, np.zeros(3)


def _read_cube(filecontent):
    textbuf = io.StringIO(filecontent.decode())
    cube = ase.io.cube.read_cube(textbuf)
    # sometimes there is an entry at index 3
    # denoting the number of values for each grid position
    origin = cube["origin"][0:3]
    # by convention the cube electron density is given in electrons/Bohr^3,
    # and ase read_cube does not convert to electrons/Å^3, so we do the conversion here
    cube["data"] *= 1.0 / ase.units.Bohr ** 3
    return cube["data"], cube["atoms"], origin
