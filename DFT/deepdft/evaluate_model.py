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
"""evaluate_model"""
import argparse
import os
import tarfile
import logging
import timeit

import numpy as np
import mindspore as ms
from   mindspore import ops, context

import src.dataset as dataset
import src.densitymodel as densitymodel
from   src.utils import write_cube_to_tar, load_cfg
from   train import split_data


def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(
        description="Evaluate density model", fromfile_prefix_chars="@"
    )
    parser.add_argument("--config", type=str, default="configs/config_eval.yaml", help="Config file")
    return load_cfg(parser.parse_args(arg_list).config)

def batch_map_fn(data):
    batch_data = {}
    for key, value in data[0].items():
        if isinstance(value, ms.Tensor):
            batch_data[key] = value.asnumpy()
        elif isinstance(value, dict):
            batch_data[key] = {"filename": str(value["filename"])}
        else:
            batch_data[key] = value
    return batch_data

def main():
    args = get_arguments()

    # Setup logging
    os.makedirs(args.output_dir, exist_ok=True)
    handlers = [
        logging.FileHandler(
            os.path.join(args.output_dir, "printlog.txt"), mode="w"
        ),
        logging.StreamHandler(),
    ]

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=handlers,
    )

    logging.debug("pid: %d", os.getpid())
    context.set_context(mode=context.GRAPH_MODE if args.mode.upper().startswith("GRAPH") else context.PYNATIVE_MODE,
                        device_target=args.device_target, 
                        device_id=args.device_id)
    logging.debug("Running in %s mode, using device %s", context.get_context("mode"), context.get_context("device_target"))

    if args.ignore_pbc and args.force_pbc:
        raise ValueError("ignore_pbc and force_pbc are mutually exclusive and can't both be set at the same time")
    elif args.ignore_pbc:
        set_pbc = False
    elif args.force_pbc:
        set_pbc = True
    else:
        set_pbc = None

    # Initialise model and load model
    net = densitymodel.PainnDensityModel(args.num_interactions, args.node_size, args.cutoff,)
    logging.info("loading model from %s", args.load_model)

    state_dict = ms.load_checkpoint(args.load_model)
    ms.load_param_into_net(net, state_dict)
    net.set_train(False)

    # Load dataset
    if args.dataset.endswith(".txt"):
        # Text file contains list of datafiles
        with open(args.dataset, "r") as datasetfiles:
            filelist = [os.path.join(os.path.dirname(args.dataset), line.strip('\n')) for line in datasetfiles]
    else:
        filelist = [args.dataset]

    logging.info("loading data %s", args.dataset)
    densitydata = None
    for path in filelist:
        if not densitydata:
            densitydata = dataset.DensityData(path)
        else:
            densitydata.concat(dataset.DensityData(path))

    # Split data into train and validation sets
    if args.split_file:
        datasplits = split_data(densitydata, args)
    else:
        datasplits = {"all": densitydata}

    for split_name, densitydataset in datasplits.items():
        if args.split and split_name not in args.split:
            continue

        logging.info("Test for %s with %d entries", split_name, len(densitydataset))

        if args.write_error_cubes:
            outname = os.path.join(args.output_dir, "eval_" + split_name + ".tar")
            tar = tarfile.open(outname, "w")

        for dict_idx in range(len(densitydataset)):
            density = []
            density_dict = densitydataset[dict_idx]

            # Loop over all slices
            density_iter = dataset.DensityGridIterator(density_dict, args.probe_count, args.cutoff, set_pbc)

            # Make graph with no probes
            collate_fn = dataset.CollateFuncAtoms(
                cutoff=args.cutoff,
                set_pbc_to=set_pbc,
            )
            device_batch = collate_fn([density_dict])
            device_batch = dataset.collate_list_of_dicts(device_batch, None)

            atom_representation_scalar, atom_representation_vector = net.atom_model(device_batch)

            num_positions = np.prod(density_dict["grid_position"].shape[0:3])
            sum_abs_error     = 0.
            sum_squared_error = 0.
            sum_target        = 0.

            for slice_id, probe_graph_dict in enumerate(density_iter):
                # Transfer target to device
                flat_index   = np.arange(slice_id*args.probe_count, min((slice_id+1)*args.probe_count, num_positions))
                pos_index    = np.unravel_index(flat_index, density_dict["density"].shape[0:3])
                probe_target = density_dict["density"][pos_index]

                # Transfer model input to device
                probe_dict = dataset.collate_list_of_dicts_ms([probe_graph_dict])
                device_batch["probe_edges"]              =  probe_dict["probe_edges"]
                device_batch["probe_edges_displacement"] =  probe_dict["probe_edges_displacement"]
                device_batch["probe_xyz"]                =  probe_dict["probe_xyz"]
                device_batch["num_probe_edges"]          =  probe_dict["num_probe_edges"]
                device_batch["num_probes"]               =  probe_dict["num_probes"]

                res = net.probe_model(device_batch, atom_representation_scalar, atom_representation_vector)

                # Compare result with target
                error = probe_target - res.asnumpy()
                sum_abs_error     += np.sum(np.abs(error))
                sum_squared_error += np.sum(np.square(error))
                sum_target        += np.sum(probe_target)

                if args.write_error_cubes:
                    density.append(res.numpy())

            voxel_volume = density_dict["atoms"].get_volume()/np.prod(density_dict["density"].shape)
            rmse = np.sqrt((sum_squared_error/num_positions))
            mae  = sum_abs_error/num_positions
            abserror_integral = sum_abs_error*voxel_volume
            total_integral = sum_target*voxel_volume
            percentage_error = 100*abserror_integral/total_integral


            if args.write_error_cubes:
                pred_density = np.concatenate(density, axis=1)
                target_density = density_dict["density"]
                pred_density = pred_density.reshape(target_density.shape)
                errors = pred_density-target_density

                fname_stripped = density_dict["metadata"]["filename"]
                while fname_stripped.endswith(".zz"):
                    fname_stripped = fname_stripped[:-3]
                name, _ = os.path.splitext(fname_stripped)
                write_cube_to_tar(
                    tar,
                    density_dict["atoms"],
                    pred_density,
                    density_dict["grid_position"][0, 0, 0],
                    name + "_prediction" + ".cube" + ".zz",
                    )
                write_cube_to_tar(
                    tar,
                    density_dict["atoms"],
                    errors,
                    density_dict["grid_position"][0, 0, 0],
                    name + "_error" + ".cube" + ".zz",
                    )
                write_cube_to_tar(
                    tar,
                    density_dict["atoms"],
                    target_density,
                    density_dict["grid_position"][0, 0, 0],
                    name + "_target" + ".cube" + ".zz",
                    )

            logging.info("split=%s, filename=%s, mae=%f, rmse=%f, abs_relative_error=%f%%", split_name, density_dict["metadata"]["filename"], mae, rmse, percentage_error)


if __name__ == "__main__":
    main()
