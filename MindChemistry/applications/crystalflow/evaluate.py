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
"""evaluate file"""
import os
import pickle
import time
import argparse
import yaml

import mindspore as ms

from data.crysloader import Crysloader
from data.dataset import fullconnect_dataset
from models.cspnet import CSPNet
from models.flow import CSPFlow
from models.infer_utils import (count_consecutive_occurrences,
                                lattices_to_params_ms)


def flow(loader, model, num_evals, n, anneal_slope, anneal_offset):
    """Generating "num_evals" crystals for each composition in the dataset.

    Args:
        loader (Crysloader): The dataset loader.
        model (nn.cell): The diffution model.
        num_evals (int): The number of generated crystals for each composition
        step_lr (float): Langevin dynamics. Defaults to 1e-5.

    Returns:
        Tuple(List[Dict], List[List[Dict]]): The ground truth and predicted crystals.The form is
           as follows:
            ...
            (
                [
                        [Crystal A sample 1, Crystal A sample 2, Crystal A sample 3, ... Crystal A sample num_eval],
                        [Crystal B sample 1, Crystal B sample 2, Crystal B sample 3, ... Crystal B sample num_eval]
                        ...
                ],

                [
                        Crystal A ground truth,
                        Crystal B ground truth,
                        ...
                ]
            )
            ...
    """
    gt_struc = []
    pred_struc = []
    for atom_types_step, frac_coords_step, _, lengths_step, \
        angles_step, _, num_atoms_step, edge_index_step, node_batch_step, \
        node_mask_step, edge_mask_step, batch_mask, _, batch_size_step in loader:
        num_node_list = count_consecutive_occurrences(
            node_batch_step.asnumpy().tolist())

        pred_struc_batch = [[] for _ in range(batch_size_step)]
        epoch_starttime = time.time()
        for eval_idx in range(num_evals):
            print(
                f'Batch {loader.step} / {loader.step_num+1}, sample {eval_idx} / {num_evals}'
            )
            starttime = time.time()
            _, frac_coords_t, lattices_t = model.sample(node_batch_step,
                                                        node_mask_step,
                                                        edge_mask_step,
                                                        batch_mask,
                                                        atom_types_step,
                                                        edge_index_step,
                                                        num_atoms_step,
                                                        n,
                                                        anneal_slope,
                                                        anneal_offset)
            lengths_pred, angles_pred = lattices_to_params_ms(
                lattices_t[:batch_size_step])

            start_index = 0
            for i in range(batch_size_step):
                num_node_i = num_node_list[i]
                atom_types_i = atom_types_step[start_index:start_index +
                                               num_node_i].asnumpy()
                frac_coords_i = frac_coords_t[start_index:start_index +
                                              num_node_i].asnumpy()
                lengths_i = lengths_pred[i].asnumpy()
                angles_i = angles_pred[i].asnumpy()

                pred_struc_batch[i].append({
                    'atom_types': atom_types_i,
                    'frac_coords': frac_coords_i,
                    'lengths': lengths_i,
                    'angles': angles_i
                })

                if eval_idx == 0:
                    frac_coords_gt = frac_coords_step[start_index:start_index +
                                                      num_node_i].asnumpy()
                    lengths_gt = lengths_step[i].asnumpy()
                    angles_gt = angles_step[i].asnumpy()
                    gt_struc.append({
                        'atom_types': atom_types_i,
                        'frac_coords': frac_coords_gt,
                        'lengths': lengths_gt,
                        'angles': angles_gt
                    })

                start_index += num_node_i

            starttime0 = starttime
            starttime = time.time()
            print(f"Evaluation time: {starttime - starttime0} s")
        pred_struc.extend(pred_struc_batch)

        print(
            f"##########Evaluation time for one Batch : \
            {time.time() - epoch_starttime} s ################"
        )
    return gt_struc, pred_struc

def main(args):
    """main
    """
    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)

    test_datatset = fullconnect_dataset(
        name=config['dataset']['data_name'],
        path=config['dataset']['test']['path'],
        save_path=config['dataset']['test']['save_path'])
    test_loader = Crysloader(config['test']['batch_size'],
                             *test_datatset,
                             shuffle_dataset=False)

    decoder = CSPNet(num_layers=config['model']['num_layers'],
                     hidden_dim=config['model']['hidden_dim'],
                     num_freqs=config['model']['num_freqs'])
    mindspore_ckpt = ms.load_checkpoint(config['checkpoint']['last_path'])
    ms.load_param_into_net(decoder, mindspore_ckpt)

    model = CSPFlow(decoder)

    model.set_train(False)

    gt_struc, pred_struc = flow(test_loader,
                                model,
                                config['test']['num_eval'],
                                n=args.N,
                                anneal_slope=args.anneal_slope,
                                anneal_offset=args.anneal_offset)

    eval_save_path = config['test']['eval_save_path']
    os.makedirs(os.path.dirname(eval_save_path), exist_ok=True)

    with open(eval_save_path, 'wb') as f:
        pickle.dump({'pred': pred_struc, 'gt': gt_struc}, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('-N', type=int, default=1000)
    parser.add_argument('--anneal_slope', type=float, default=0.0)
    parser.add_argument('--anneal_offset', type=float, default=0.0)
    main_args = parser.parse_args()
    main(main_args)
    