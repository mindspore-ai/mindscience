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
"""predict_with_model"""
import logging
import os
import json
import argparse
import math
import contextlib
import timeit

import ase
import ase.io
import numpy as np
import mindspore as ms
from   mindspore import ops, context

import src.dataset as dataset
import src.densitymodel as densitymodel
import src.utils as utils

def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(
        description="Predict with pretrained model", fromfile_prefix_chars="@"
    )
    parser.add_argument("--config", type=str, default="configs/config_pred.yaml", help="Config file")
    return utils.load_cfg(parser.parse_args(arg_list).config)

def load_model(model_dir):
    with open(os.path.join(model_dir, "arguments.json"), "r") as f:
        runner_args = argparse.Namespace(**json.load(f))
    model = densitymodel.PainnDensityModel(runner_args.num_interactions, runner_args.node_size, runner_args.cutoff)
    param_dict = ms.load_checkpoint(os.path.join(model_dir, "best_model.ckpt"))
    param_not_load, _ = ms.load_param_into_net(model, param_dict)
    return model, runner_args.cutoff

class LazyMeshGrid():
    def __init__(self, cell, grid_step, origin=None, adjust_grid_step=False):
        self.cell = cell
        if adjust_grid_step:
            n_steps = np.round(self.cell.lengths()/grid_step)
            self.scaled_grid_vectors = [np.arange(n)/n for n in n_steps]
            self.adjusted_grid_step = self.cell.lengths()/n_steps
        else:
            self.scaled_grid_vectors = [np.arange(0, l, grid_step)/l for l in self.cell.lengths()]
        self.shape = np.array([len(g) for g in self.scaled_grid_vectors] + [3])
        if origin is None:
            self.origin = np.zeros(3)
        else:
            self.origin = origin

        self.origin = np.expand_dims(self.origin, 0)

    def __getitem__(self, indices):
        indices = np.array(indices)
        indices_shape = indices.shape
        if not (len(indices_shape) == 2 and indices_shape[0] == 3):
            raise NotImplementedError("Indexing must be a 3xN array-like object")
        gridA = self.scaled_grid_vectors[0][indices[0]]
        gridB = self.scaled_grid_vectors[1][indices[1]]
        gridC = self.scaled_grid_vectors[2][indices[2]]

        grid_pos = np.stack([gridA, gridB, gridC], 1)
        grid_pos = np.dot(grid_pos, self.cell)
        grid_pos += self.origin

        return grid_pos


def ceil_float(x, step_size):
    # Round up to nearest step_size and subtract a small epsilon
    x = math.ceil(x/step_size) * step_size
    eps = 2*np.finfo(float).eps * x
    return x - eps

def load_atoms(atomspath, vacuum, grid_step):
    atoms = ase.io.read(atomspath)

    if np.any(atoms.get_pbc()):
        atoms, grid_pos, origin = load_material(atoms, grid_step)
    else:
        atoms, grid_pos, origin = load_molecule(atoms, grid_step, vacuum)

    metadata = {"filename": atomspath}
    res = {
        "atoms": atoms,
        "origin": origin,
        "grid_position": grid_pos,
        "metadat": metadata,
    }

    return res

def load_material(atoms, grid_step):
    atoms = atoms.copy()
    grid_pos = LazyMeshGrid(atoms.get_cell(), grid_step, adjust_grid_step=True)
    origin = np.zeros(3)

    return atoms, grid_pos, origin

def load_molecule(atoms, grid_step, vacuum):
    atoms = atoms.copy()
    atoms.center(vacuum=vacuum) # This will create a cell around the atoms

    # Readjust cell lengths to be a multiple of grid_step
    a, b, c, ang_bc, ang_ac, ang_ab = atoms.get_cell_lengths_and_angles()
    a, b, c = ceil_float(a, grid_step), ceil_float(b, grid_step), ceil_float(c, grid_step)
    atoms.set_cell([a, b, c, ang_bc, ang_ac, ang_ab])

    origin = np.zeros(3)

    grid_pos = LazyMeshGrid(atoms.get_cell(), grid_step)

    return atoms, grid_pos, origin

def main():
    args = get_arguments()

    # Setup logging
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(
                os.path.join(args.output_dir, "printlog.txt"), mode="w"
            ),
            logging.StreamHandler(),
        ],
    )

    logging.debug("pid: %d", os.getpid())
    context.set_context(mode=context.GRAPH_MODE if args.mode.upper().startswith("GRAPH") else context.PYNATIVE_MODE,
                        device_target=args.device_target, 
                        device_id=args.device_id)
    logging.debug("Running in %s mode, using device %s", context.get_context("mode"), context.get_context("device_target"))

    model, cutoff = load_model(args.model_dir)
    model.set_train(False)
    density_dict  = load_atoms(args.atoms_file, args.vacuum, args.grid_step)

    cubewriter = utils.CubeWriter(
        os.path.join(args.output_dir, "prediction.cube"),
        density_dict["atoms"],
        density_dict["grid_position"].shape[0:3],
        density_dict["origin"],
        "predicted by DeepDFT model",
    )

    if args.ignore_pbc and args.force_pbc:
        raise ValueError("ignore_pbc and force_pbc are mutually exclusive and can't both be set at the same time")
    elif args.ignore_pbc:
        set_pbc = False
    elif args.force_pbc:
        set_pbc = True
    else:
        set_pbc = None

    start_time = timeit.default_timer()

    contextmanager = contextlib.nullcontext()
    with contextmanager:
        # Make graph with no probes
        logging.debug("Computing atom-to-atom graph")
        collate_fn = dataset.CollateFuncAtoms(
            cutoff=cutoff,
            set_pbc_to=set_pbc,
        )
        graph_dict = collate_fn([density_dict])
        graph_dict = dataset.collate_list_of_dicts(graph_dict, None)
        logging.debug("Computing atom representation")

        device_batch = graph_dict

        device_batch['nodes'] = device_batch['nodes'].astype(ms.int32)
        device_batch['num_nodes'] = device_batch['num_nodes'].astype(ms.int32)
        device_batch['num_atom_edges'] = device_batch['num_atom_edges'].astype(ms.int32)

        atom_representation_scalar, atom_representation_vector = model.atom_model(device_batch)
        logging.debug("Atom representation done")

        # Loop over all slices
        density_iter = dataset.DensityGridIterator(density_dict, args.probe_count, cutoff, set_pbc_to=set_pbc)
        density = []
        for probe_graph_dict in density_iter:
            probe_dict = dataset.collate_list_of_dicts_ms([probe_graph_dict])
            device_batch["probe_edges"]              = probe_dict["probe_edges"]
            device_batch["probe_edges_displacement"] = probe_dict["probe_edges_displacement"]
            device_batch["probe_xyz"]                = probe_dict["probe_xyz"]
            device_batch["num_probe_edges"]          = probe_dict["num_probe_edges"].astype(ms.int32)
            device_batch["num_probes"]               = probe_dict["num_probes"].astype(ms.int32)

            epoch_start_time = timeit.default_timer()
            res = model.probe_model(
                device_batch, atom_representation_scalar, atom_representation_vector,
            )
            epoch_end_time = timeit.default_timer()
            
            density = res

            cubewriter.write(density.numpy().flatten())
            logging.debug("Written %d/%d, per_step_time=%f s", cubewriter.numbers_written,
                          np.prod(density_dict["grid_position"].shape[0:3]), epoch_end_time - epoch_start_time)

    end_time = timeit.default_timer()
    logging.info("done time_elapsed=%f", end_time-start_time)

if __name__ == "__main__":
    main()
