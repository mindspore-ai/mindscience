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
User-defined wrapper for training and testing
"""
import os
import os.path
import stat
import time
import logging
import mindspore as ms
import numpy as np
from skimage.transform import warp_polar
from scipy.fft import fftn, fftshift
from src.module import GaussianProcessTensors
from src import utils as du


class DensityFunctionalTheory:
    """Class of DensityFunctionalTheory"""

    def __init__(self, data_dir='.', workspace_dir='.', train_dir='water_102', test_dir='water_102', train_number=50,
                 test_number=50, cv_indices=None, train_indices_file=None, test_indices_file=None, cv_indices_file=None,
                 run_id=1, output_file=None, energy_type='cc', descriptor_type='pot', gaussian_width=0.6,
                 grid_spacing=0.3, verbose=0, density_kernel_type='rbf', energy_kernel_type='rbf', grid_file=None,
                 use_true_densities=False, importance_weighting=False, invariance=False, max_iter=None,
                 n_training=None, norm_params=None):
        """init"""
        self.data_dir = data_dir
        self.workspace_dir = workspace_dir
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.train_indices = list(range(0, train_number))
        self.test_indices = list(range(train_number + 1, train_number + 1 + test_number))
        self.cv_indices = cv_indices
        self.train_indices_file = train_indices_file
        self.test_indices_file = test_indices_file
        self.cv_indices_file = cv_indices_file
        self.run_id = run_id
        self.output_file = output_file
        self.energy_type = energy_type
        self.descriptor_type = descriptor_type
        self.gaussian_width = gaussian_width
        self.grid_spacing = grid_spacing
        self.grid_file = grid_file
        self.grid_steps = None
        self.invariance = invariance
        self.max_iter = max_iter
        self.use_true_densities = use_true_densities
        self.verbose = verbose
        self.importance_weighting = importance_weighting
        self.n_training = n_training
        self.density_kernel = density_kernel_type
        self.energy_kernel = energy_kernel_type
        self.density_kernel_params = {}
        self.energy_kernel_params = {}
        self.norm_params = norm_params

    def _check_indices(self):
        """check_indices"""
        if self.train_indices_file is not None:
            self.train_indices = list(np.load(os.path.join(self.workspace_dir, self.train_indices_file)))

        if self.cv_indices_file is not None:
            self.cv_indices = np.load(self.workspace_dir, self.cv_indices_file)

    def _set_importance_weights(self):
        """ set importance weights"""
        if self.importance_weighting:
            importance_weights = np.load(os.path.join(self.workspace_dir, self.train_dir, 'importance_weights_' +
                                                      self.descriptor_type + '_' + str(self.run_id) + '.npy'))
            importance_weights = ms.Tesnor.from_numpy(importance_weights[self.train_indices])
        else:
            importance_weights = None
        return importance_weights

    def _write_error(self, file, energy, energy_pred):
        """ save and write error"""
        file.write('训练采样点:' + str(self.test_indices) + '\n')
        file.write('能量:\n')
        corr = np.corrcoef(energy.asnumpy().T, energy_pred.asnumpy().T)[0][1]
        if np.isnan(corr):
            corr = 0
        if self.verbose > 0:
            logging.info("相关系数等于: %s", corr)
        file.write(f'相关系数(CC): {corr}\n')
        file.write(f'均方根误差(RootMSE): {(energy - energy_pred).pow(2).mean(axis=0).mean().sqrt().asnumpy()}\n')
        file.write(
            f'平均绝对误差(MAE): {(energy.astype(ms.float32) - energy_pred).abs().mean(axis=0).mean().asnumpy()}\n')
        file.write(f'最大绝对误差(Max MAE): {(energy.astype(ms.float32) - energy_pred).abs().max().asnumpy()}\n')

    def calculate_potentials(self):
        """calculate the sum of artificial Gaussian functions as the potential field"""
        times = time.time()
        if self.verbose >= 0:
            logging.info('计算人工电势')
        if self.train_dir == self.test_dir:
            work_dirs = [self.train_dir]
        else:
            work_dirs = [self.train_dir, self.test_dir]

        all_positions = []
        for work_dir in work_dirs:
            positions = np.load(os.path.join(self.data_dir, work_dir, 'structures.npy')).astype(np.float32)  # 读取分子结构
            all_positions.append(positions)
        cat_positions = np.concatenate(all_positions, axis=0)
        if self.verbose > 0:
            logging.info("位置形状: %s", cat_positions.shape)
        max_pos, min_pos, max_range = du.calculate_positions(cat_positions)
        if self.verbose > 0:
            logging.info("最小/最大位置坐标: %s, %s", min_pos, max_pos)

        steps = (max_range / self.grid_spacing).round().astype(np.int64)
        self.grid_steps = steps
        if self.grid_file is None:
            grid_range = [np.linspace(ma, mm, s) for ma, mm, s in zip(max_pos, min_pos, steps)]
        elif self.grid_file == '':
            if self.verbose > 0:
                logging.info('保存电势网格')
            grid_range = [np.linspace(ma, mm, s) for ma, mm, s in zip(max_pos, min_pos, steps)]
            grid_range = np.array(grid_range, dtype=object)
            np.save(os.path.join(self.workspace_dir, self.train_dir, 'grid_range_' + str(self.run_id) + '.npy'),
                    grid_range)
        elif os.path.exists(os.path.join(self.workspace_dir, self.train_dir,
                                         self.grid_file + '_' + str(self.run_id) + '.npy')):
            grid_range = np.load(
                os.path.join(self.workspace_dir, self.train_dir, self.grid_file + '_' + str(self.run_id)
                             + '.npy'), allow_pickle=True)
        else:
            raise RuntimeError('网格文件无效: \'' + self.grid_file + '\'')
        grid = du.calculate_grid(grid_range)
        for i, work_dir in enumerate(work_dirs):
            charges = np.load(os.path.join(self.workspace_dir, work_dir, 'atom_types.npy'))  # 必填：atom_types.npy分子中原子序号
            positions = all_positions[i]
            potentials = du.calculate_potential(positions, charges, self.gaussian_width, grid)
            if self.invariance:
                potentials = self._spectral_transform(potentials)
            if self.verbose > 1:
                logging.info("potentials.shape: %s", potentials.shape)
            np.save(os.path.join(self.workspace_dir, work_dir, 'pot_' + str(self.run_id) + '.npy'), potentials)
        if self.verbose > 0:
            logging.info("电势位置形状: %s", grid.shape)
        logging.info("计算人工电势运行时长: %s", str(time.time() - times))

    def _descriptors_to_density(self):
        """
        training a model that predicts the basis function coefficients of the density from the descriptors
        """
        logging.info('---训练从描述符预测电子密度的模型---')
        self._check_indices()
        descriptors = np.load(
            os.path.join(self.workspace_dir, self.train_dir, self.descriptor_type + '_' + str(self.run_id)
                         + '.npy'))
        coefficients = np.load(os.path.join(self.data_dir, self.train_dir, 'dft_densities.npy'))
        importance_weights = self._set_importance_weights()
        descriptors = ms.Tensor.from_numpy(descriptors[self.train_indices, :].astype(np.float32))
        coefficients = ms.Tensor.from_numpy(coefficients[self.train_indices, :].astype(np.float32))
        density_kr = GaussianProcessTensors(run_id=self.run_id, verbose=self.verbose, workspace_dir=self.workspace_dir,
                                            kernel=self.density_kernel,
                                            kernel_params=self.density_kernel_params)
        start = time.time()
        density_kr.fit(descriptors, coefficients, importance_weights=importance_weights, max_iter=self.max_iter,
                       save_type='density', mean_std=self.norm_params)
        end = time.time()
        logging.info("训练用时: %s", end - start)
        density_kr.save_file(os.path.join(self.workspace_dir, self.train_dir, 'density_kr_' + str(self.run_id)))
        return np.mean(density_kr.errors)

    def _density_to_energy(self):
        """
        training a Model to predict energy from density
        """
        logging.info('---训练由密度泛函预测能量的模型---')

        if self.use_true_densities:
            coefficients_pred = np.load(os.path.join(self.data_dir, self.train_dir, 'dft_densities.npy'))
        else:
            coefficients_pred = np.load(os.path.join(self.workspace_dir, self.train_dir, 'coefficients_pred_' +
                                                     str(self.run_id) + '.npy'))
        self._check_indices()
        if self.use_true_densities:
            suffix = 'true'
        else:
            suffix = 'pred'
        if self.importance_weighting:
            importance_weights = np.load(os.path.join(self.workspace_dir, self.train_dir, 'importance_weights_dens_' +
                                                      suffix + '_' + str(self.run_id) + '.npy'))
            importance_weights = importance_weights[self.train_indices]
        else:
            importance_weights = None

        if self.energy_type == 'diff':
            energies1 = np.load(os.path.join(self.data_dir, self.train_dir, 'cc_energies.npy'))
            energies2 = np.load(os.path.join(self.data_dir, self.train_dir, 'dft_energies.npy'))
            energies = energies1 - energies2
        else:
            energies = np.load(os.path.join(self.data_dir, self.train_dir, self.energy_type + '_energies.npy'))
        energies = energies[self.train_indices]

        coefficients_pred = ms.Tensor.from_numpy(coefficients_pred[self.train_indices, :])
        energies = ms.Tensor.from_numpy(np.reshape(energies, (-1, 1)))
        if self.verbose > 1:
            logging.info(energies.shape)

        energy_kr = GaussianProcessTensors(run_id=self.run_id + .5, verbose=self.verbose, mae=True,
                                           workspace_dir=self.workspace_dir, kernel=self.energy_kernel,
                                           kernel_params=self.energy_kernel_params)
        energy_kr.fit(coefficients_pred, energies, importance_weights=importance_weights, max_iter=self.max_iter,
                      save_type='energy', mean_std=self.norm_params)

        energy_kr.save_file(os.path.join(self.workspace_dir, self.train_dir, str(self.energy_type) + str('_kr_')
                                         + str(self.run_id)))

        energy_pred = energy_kr.predict(coefficients_pred, save_type='energy', mean_std=self.norm_params)
        if self.verbose > 0:
            logging.info("能量训练误差: %s", (energies - energy_pred).abs().mean(axis=0))
        return (energies.astype(ms.float32) - energy_pred).abs().mean(axis=0)

    def _spectral_transform(self, descriptors):
        """
        projected onto a log-polar coordinate system then Fourier transformed
        """
        transformed_descriptor = []
        for idx in range(descriptors.shape[0]):
            temp = descriptors[idx, :].reshape(self.grid_steps[0], self.grid_steps[1], self.grid_steps[2])
            fft_polar = np.abs(fftshift(fftn(warp_polar(temp, channel_axis=1, scaling='log',
                                                        radius=self.grid_steps.min()))))
            transformed_descriptor.append(
                fft_polar[91:270, :self.grid_steps[1] // 2, :self.grid_steps[2] // 2].reshape(-1))

        transformed_descriptor = np.array(transformed_descriptor)
        return transformed_descriptor

    def _predict_density(self):
        """
        predicting density coefficients using the learned KRR model
        """
        if self.verbose > 0:
            logging.info('预测密度系数')
        for work_dir in [self.train_dir, self.test_dir]:
            descriptors = np.load(os.path.join(self.workspace_dir, work_dir, self.descriptor_type + '_' +
                                               str(self.run_id) + '.npy'))
            coefficients = np.load(os.path.join(self.data_dir, work_dir, 'dft_densities.npy'))
            descriptors = ms.Tensor.from_numpy(descriptors).astype(ms.float32)
            density_kr = GaussianProcessTensors(run_id=self.run_id, verbose=self.verbose,
                                                workspace_dir=self.workspace_dir)
            density_kr.load_file(os.path.join(self.workspace_dir, self.train_dir, 'density_kr_' + str(self.run_id)))
            coefficients_pred = density_kr.predict(descriptors, save_type='density',
                                                   mean_std=self.norm_params)

            if self.verbose > 0:
                logging.info(
                    "均方根误差(RootMSE): %s",
                    np.mean(np.linalg.norm(coefficients[:len(coefficients_pred)] - coefficients_pred.asnumpy(), axis=1))
                )
                logging.info(
                    "系数范数: %s",
                    np.mean(np.linalg.norm(coefficients, axis=1))
                )
                logging.info(
                    "预测系数范数: %s",
                    np.mean(np.linalg.norm(coefficients_pred, axis=1))
                )
            np.save(os.path.join(self.workspace_dir, work_dir, 'coefficients_pred_' + str(self.run_id) + '.npy'),
                    coefficients_pred.asnumpy())
        return np.mean(np.linalg.norm(coefficients[:len(coefficients_pred)] - coefficients_pred.asnumpy(), axis=1))

    def test(self):
        """
        evaluate the model using the data in id_range
        """
        logging.info('评估模型')
        if self.verbose > 0:
            logging.info('开始测试')

        if self.test_indices_file is not None:
            self.test_indices = list(np.load(os.path.join(self.workspace_dir, self.test_indices_file)))
        cc_energies = np.load(os.path.join(self.data_dir, self.train_dir, 'cc_energies.npy'))
        dft_energies = np.load(os.path.join(self.data_dir, self.train_dir, 'dft_energies.npy'))

        if self.use_true_densities:
            coefficients_pred = np.load(os.path.join(self.data_dir, self.test_dir, 'dft_densities.npy'))
        else:
            coefficients_pred = np.load(os.path.join(self.workspace_dir, self.test_dir, 'coefficients_pred_' +
                                                     str(self.run_id) + '.npy'))

        coefficients_pred = ms.Tensor.from_numpy(coefficients_pred[self.test_indices, :])
        cc_energies = cc_energies[self.test_indices]
        cc_energies = ms.Tensor.from_numpy(np.reshape(cc_energies, (-1, 1))).astype(ms.float32)
        dft_energies = dft_energies[self.test_indices]
        dft_energies = ms.Tensor.from_numpy(np.reshape(dft_energies, (-1, 1)))

        if self.verbose > 0:
            logging.info('加载能量核')
        energy_kr = GaussianProcessTensors(run_id=self.run_id + .5, verbose=self.verbose,
                                           workspace_dir=self.workspace_dir)
        energy_kr.load_file(
            os.path.join(self.workspace_dir, self.train_dir, self.energy_type + '_kr_' + str(self.run_id)))

        if self.verbose > 0:
            logging.info('预测能量')
        if self.energy_type == 'diff':
            energies_pred = (
                energy_kr.predict(coefficients_pred, save_type='energy',
                                  mean_std=self.norm_params).reshape(-1, 1) + dft_energies).astype(ms.float32)
        else:
            energies_pred = energy_kr.predict(coefficients_pred, save_type='energy',
                                              mean_std=self.norm_params).reshape(-1, 1).astype(ms.float32)

        if self.energy_type == 'dft':
            errors_pred = (cc_energies.min() - energies_pred).abs()  # check dimension
        else:
            errors_pred = (cc_energies.astype(ms.float32) - energies_pred).abs()
        du.write_error(energy=cc_energies, energy_pred=energies_pred)
        if self.output_file is not None:
            if self.verbose > 0:
                logging.info(
                    "写入文件: %s",
                    os.path.join(self.workspace_dir, self.test_dir,
                                 self.output_file + '_' + self.energy_type + '_' + str(self.run_id) + '.npy')
                )
            with os.fdopen(os.open(os.path.join(self.workspace_dir, self.test_dir,
                                                self.output_file + '_' + self.energy_type + '_' + str(
                                                    self.run_id) + '.npy'), os.O_WRONLY | os.O_CREAT, stat.S_IWUSR),
                           'w') as f:
                self._write_error(file=f, energy=cc_energies, energy_pred=energies_pred)
            np.save(os.path.join(self.workspace_dir, self.test_dir, 'errors_pred_' + self.energy_type + '_' +
                                 str(self.run_id) + '.npy'), errors_pred.asnumpy())

        return (cc_energies.astype(ms.float32) - energies_pred).abs().mean(axis=0)

    def train(self):
        """
        perform the entire training process to predict the energy
        """
        logging.info('训练开始')
        dens_cv_err = self._descriptors_to_density()
        dens_err = self._predict_density()
        en_err = self._density_to_energy()
        return dens_cv_err, dens_err, en_err
