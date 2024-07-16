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
# ==============================================================================
"""
train
"""
import time
import gc
import argparse
import os
import io
import logging
import yaml
import mindspore as ms
import numpy as np
from src.trainer import DensityFunctionalTheory
from src.utils import calculate_l1_error, save_mae_chart, save_error_chart

if __name__ == '__main__':
    LOG_FORMAT = "时间：%(asctime)s - 日志等级：%(levelname)s - 日志信息：%(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('-max_iter', type=int, default=5000, help="max_iter")
    parser.add_argument("-mode", default=ms.GRAPH_MODE)
    parser.add_argument("-device_target", type=str, default='GPU', choices=["GPU", "Ascend"])
    parser.add_argument("-device_id", type=int, default=0)
    parser.add_argument("-config_path", type=str, default=os.path.abspath('.') + '/config.yml')
    args = parser.parse_args()

    # load config
    with io.open(args.config_path, 'r') as stream:
        params = yaml.safe_load(stream)
    ensem_params = params['ensem_params']
    ensem_params['max_iter'] = args.max_iter
    energy_types = ensem_params['energy_types']
    charges = np.array(ensem_params['charges'])
    n_trainings = ensem_params['n_trainings']
    n_test = ensem_params['n_test']
    data_dir = ensem_params['data_dir']
    train_dir = ensem_params['train_dir']
    test_dir = ensem_params['test_dir']
    gaussian_width = ensem_params['gaussian_width']
    spacing = ensem_params['spacing']
    metric = ensem_params['metric']
    invariance = ensem_params['invariance']
    opt_type = params['train_params']['opt_type']
    verbose = ensem_params['verbose']
    workspace_dir = ensem_params['workspace_dir']
    density_kernel_type = ensem_params['density_kernel_type']
    energy_kernel_type = ensem_params['energy_kernel_type']
    norm_params = params['norm_params']

    max_iter = args.max_iter
    train_indices_file = 'train_indices_file.npy'
    test_indices_file = 'test_indices_file.npy'

    # create save directory for current experiment
    if not os.path.isdir(workspace_dir):
        os.mkdir(workspace_dir)

    # set context
    ms.set_context(mode=args.mode, device_target=args.device_target, device_id=args.device_id)

    # Start training
    logging.info("开始")
    times = time.time()
    maes_lists = []
    coefficients_errors_lists = []
    work_dirs = [train_dir, test_dir] if train_dir != test_dir else [train_dir]
    for w_dir in work_dirs:
        np.save(os.path.join(workspace_dir, w_dir, 'atom_types.npy'), charges)
    train_size = np.load(os.path.join(data_dir, train_dir, 'cc_energies.npy')).size
    test_size = 1000 if 'phenol' in test_dir else np.load(os.path.join(data_dir, test_dir, 'cc_energies.npy')).size
    for energy_type in energy_types:
        maes_list = []
        coefficients_errors_list = []
        for run_id, n_training in enumerate(n_trainings):
            coefficients_errors = []
            if train_dir != test_dir:
                train_indices = np.random.choice(np.arange(train_size), n_training, replace=False)
                test_indices = np.random.choice(np.arange(test_size), n_test, replace=False)
                np.save(os.path.join(workspace_dir, train_indices_file), train_indices)
                np.save(os.path.join(workspace_dir, test_indices_file), test_indices)
            else:
                indices = list(range(n_training + n_test))
                train_indices = np.random.choice(indices, n_training, replace=False)
                test_indices = np.setdiff1d(indices, train_indices)
                np.save(os.path.join(workspace_dir, train_indices_file), train_indices)
                np.save(os.path.join(workspace_dir, test_indices_file), test_indices)
            TestDFT = DensityFunctionalTheory(data_dir=data_dir, workspace_dir=workspace_dir, train_dir=train_dir,
                                              test_dir=test_dir, run_id=run_id, train_indices_file=train_indices_file,
                                              gaussian_width=gaussian_width, test_indices_file=test_indices_file,
                                              grid_spacing=spacing, verbose=verbose,
                                              density_kernel_type=density_kernel_type, invariance=invariance,
                                              energy_kernel_type=energy_kernel_type, norm_params=norm_params,
                                              use_true_densities=False, grid_file='', energy_type=energy_type,
                                              output_file='out', max_iter=max_iter, n_training=n_training)
            TestDFT.calculate_potentials()
            TestDFT.train()
            TestDFT.test()
            gc.collect()
            coefficients_error, maes = calculate_l1_error(workspace_dir, test_dir, energy_type, run_id,
                                                          data_dir, metric)
            coefficients_errors.append(coefficients_error)
            maes_list.append(maes)
            coefficients_errors_list.append(coefficients_errors)
        maes_lists.append(maes_list)
        logging.info("mae: %s", maes_lists)
        coefficients_errors_lists.append(coefficients_errors_list)
        logging.info("coefficients errors: %s", coefficients_errors_lists)
        save_mae_chart(energy_types, maes_lists, n_trainings, workspace_dir, opt_type, max_iter)
        save_error_chart(energy_types, coefficients_errors_lists, n_trainings, workspace_dir, opt_type, max_iter,
                         metric)
    logging.info('总运行时长: %s', {time.time() - times})
