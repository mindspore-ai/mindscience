# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Evaluation
"""

import os
import time
import logging
from types import SimpleNamespace
import argparse
import mindspore as ms
import numpy as np

from mindchemistry.cell.cdvae import CDVAE
from src.dataloader import DataLoaderBaseCDVAE
from src.evaluate_utils import (get_reconstructon_res, get_generation_res,
                                get_optimization_res)
from train import get_scaler


def task_reconstruction(model, ld_kwargs, graph_dataset, recon_args):
    """Evaluate model on the reconstruction task."""
    logging.info("Evaluate model on the reconstruction task.")
    (frac_coords, num_atoms, atom_types, lengths, angles,
     gt_frac_coords, gt_num_atoms, gt_atom_types,
     gt_lengths, gt_angles) = get_reconstructon_res(
         graph_dataset, model, ld_kwargs, recon_args.num_evals,
         recon_args.force_num_atoms, recon_args.force_atom_types)

    if recon_args.label == "":
        recon_out_name = "eval_recon.npy"
    else:
        recon_out_name = f"eval_recon_{recon_args.label}.npy"

    result = {
        "eval_setting": recon_args,
        "frac_coords": frac_coords,
        "num_atoms": num_atoms,
        "atom_types": atom_types,
        "lengths": lengths,
        "angles": angles,
    }
    # save result as numpy
    np.save("./eval_result/" + recon_out_name, result)
    groundtruth = {
        "frac_coords": gt_frac_coords,
        "num_atoms": gt_num_atoms,
        "atom_types": gt_atom_types,
        "lengths": gt_lengths,
        "angles": gt_angles,
    }
    # save ground truth as numpy
    np.save("./eval_result/gt_recon.npy", groundtruth)


def task_generation(model, ld_kwargs, gen_args):
    """Evaluate model on the generation task."""
    logging.info("Evaluate model on the generation task.")

    (frac_coords, num_atoms, atom_types, lengths, angles,
     all_frac_coords_stack, all_atom_types_stack) = get_generation_res(
         model, ld_kwargs, gen_args.num_batches_to_samples, gen_args.num_evals,
         gen_args.batch_size, gen_args.down_sample_traj_step)

    if gen_args.label == "":
        gen_out_name = "eval_gen.npy"
    else:
        gen_out_name = f"eval_gen_{gen_args.label}.npy"

    result = {
        "eval_setting": gen_args,
        "frac_coords": frac_coords,
        "num_atoms": num_atoms,
        "atom_types": atom_types,
        "lengths": lengths,
        "angles": angles,
        "all_frac_coords_stack": all_frac_coords_stack,
        "all_atom_types_stack": all_atom_types_stack,
    }
    # save result as numpy
    np.save("./eval_result/" + gen_out_name, result)


def task_optimization(model, ld_kwargs, graph_dataset, opt_args):
    """Evaluate model on the property optimization task."""
    logging.info("Evaluate model on the property optimization task.")
    if opt_args.start_from == "data":
        loader = graph_dataset
    else:
        loader = None
    optimized_crystals = get_optimization_res(model, ld_kwargs, loader)
    if opt_args.label == "":
        gen_out_name = "eval_opt.npy"
    else:
        gen_out_name = f"eval_opt_{opt_args.label}.npy"
    # save result as numpy
    np.save("./eval_result/" + gen_out_name, optimized_crystals)


def main(args):
    # check whether path exists, if not exists create the direction
    folder_path = os.path.dirname(args.model_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        logging.info("%s has been created", folder_path)
    result_path = "./eval_result/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
        logging.info("%s has been created", result_path)
    config_path = "./conf/configs.yaml"
    data_config_path = f"./conf/data/{args.dataset}.yaml"
    # load model
    model = CDVAE(config_path, data_config_path)
    # load mindspore check point
    param_dict = ms.load_checkpoint(args.model_path)
    param_not_load, _ = ms.load_param_into_net(model, param_dict)
    logging.info("parameter not load: %s.", param_not_load)
    model.set_train(False)

    ld_kwargs = SimpleNamespace(n_step_each=args.n_step_each,
                                step_lr=args.step_lr,
                                min_sigma=args.min_sigma,
                                save_traj=args.save_traj,
                                disable_bar=args.disable_bar)
    # load dataset
    graph_dataset = DataLoaderBaseCDVAE(
        args.batch_size, args.dataset, shuffle_dataset=False, mode="test")
    # load scaler
    lattice_scaler, scaler = get_scaler(args)
    model.lattice_scaler = lattice_scaler
    model.scaler = scaler

    start_time_eval = time.time()
    if "recon" in args.tasks:
        task_reconstruction(model, ld_kwargs, graph_dataset, args)
    if "gen" in args.tasks:
        task_generation(model, ld_kwargs, args)
    if "opt" in args.tasks:
        task_optimization(model, ld_kwargs, graph_dataset, args)
    logging.info("end evaluation, time: %f s.", time.time() - start_time_eval)

def get_args():
    """args used for evaluation"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_target", default="Ascend", help="device target")
    parser.add_argument("--device_id", default=7, type=int, help="device id")
    parser.add_argument("--model_path", default="./loss/loss.ckpt",
                        help="path to checkpoint")
    parser.add_argument("--dataset", default="perov_5", help="name of dataset")
    parser.add_argument("--tasks", nargs="+", default=["gen"],
                        help="tasks to evaluate, choose from 'recon, gen, opt'")
    parser.add_argument("--n_step_each", default=1, type=int,
                        help="number of steps in diffusion")
    parser.add_argument("--step_lr", default=1e-3, type=float, help="learning rate")
    parser.add_argument("--min_sigma", default=0, type=float, help="minimum sigma")
    parser.add_argument("--save_traj", default=False, type=bool,
                        help="whether to save trajectory")
    parser.add_argument("--disable_bar", default=False, type=bool,
                        help="disable progress bar")
    parser.add_argument("--num_evals", default=1, type=int,
                        help="number of evaluations returned for each task")
    parser.add_argument("--num_batches_to_samples", default=1, type=int,
                        help="number of batches to sample")
    parser.add_argument("--start_from", default="data", type=str,
                        help="start from data or random")
    parser.add_argument("--batch_size", default=128, type=int, help="batch size")
    parser.add_argument("--force_num_atoms", action="store_true",
                        help="fixed num atoms or not")
    parser.add_argument("--force_atom_types", action="store_true",
                        help="fixed atom types or not")
    parser.add_argument("--down_sample_traj_step", default=10, type=int, help="down sample")
    parser.add_argument("--label", default="", help="label for output file")
    return parser.parse_args()

if __name__ == "__main__":
    main_args = get_args()
    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
    ms.context.set_context(device_target=main_args.device_target,
                           device_id=main_args.device_id, mode=1)
    main(main_args)
