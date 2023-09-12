# Copyright 2023 Huawei Technologies Co., Ltd
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
"""problem definition"""
import time

import numpy as np
import mindspore as ms
from mindspore import nn, Tensor, dataset
from mindspore.dataset import transforms

from sciai.common import TrainCellWithCallBack
from sciai.utils import print_log, amp2datatype, to_tensor

from .linear_advection_sphere import Net
from .network import Model


class Problem:
    """problem definition"""
    def __init__(self, args):
        self.data_type_str_dict = {"O0": "float32", "O1": "float16", "O2": "float16", "O3": "float16"}
        self.data_type = amp2datatype(args.amp_level)
        self.data_type_str = self.data_type_str_dict.get(args.amp_level)
        self.amp_level = args.amp_level
        self.save_ckpt_path = args.save_ckpt_path
        self.load_ckpt = args.load_ckpt
        self.load_ckpt_path = args.load_ckpt_path
        self.epochs = args.epochs
        self.layers = args.layers
        self.lr = args.lr
        self.u = args.u
        self.h = args.h
        self.nr_models = 1

        self.lmbd_left, self.lmbd_right = -np.pi, np.pi
        self.tht_lower, self.tht_upper = -np.pi / 2, np.pi / 2
        self.t0, self.t_final = 0, args.days

        self.alpha = 0 * np.pi / 2
        self.rad = 1.0 / 3
        self.lmbd0 = np.pi / 2
        self.tht0 = 0
        self.model_boundaries = [0, 4500, 9000, 13500, args.epochs]
        self.nr_boundaries = len(self.model_boundaries) - 1

        self.u00 = 2 * np.pi / args.days
        self.h00 = 1000.0

        self.alpha_ms, self.u00_ms, self.u_ms = to_tensor((self.alpha, self.u00, self.u), dtype=self.data_type)

    def u0(self, lmbd, tht):
        return self.u00 * (np.cos(tht) * np.cos(self.alpha) + np.sin(tht) * np.cos(lmbd) * np.sin(self.alpha)) / self.u

    def v0(self, lmbd):
        return - self.u00 * np.sin(lmbd) * np.sin(self.alpha) / self.u

    def r(self, lmbd, tht):
        return np.arccos(np.sin(self.tht0) * np.sin(tht) + np.cos(self.tht0) * np.cos(tht) * np.cos(lmbd - self.lmbd0))

    def h0(self, lmbd, tht):
        factor = self.h00 / 2.0 / self.h
        return factor * (1 + np.cos(np.pi * self.r(lmbd, tht) / self.rad)) * (self.r(lmbd, tht) < self.rad)

    def true_solution(self, t_p, x_p, y_p):
        return self.h0(x_p - self.u0(x_p, y_p) * t_p / np.cos(y_p), y_p - self.v0(x_p) * t_p)

    def train(self, pdes, inits):
        """train"""
        nr_batches = 0
        epoch_loss = np.zeros(self.epochs)
        self.nr_models = 1

        # initial value points
        t_init, x_init, y_init = inits[:, 0], inits[:, 1], inits[:, 2]
        h_init = self.h0(x_init, y_init)

        inits_ = np.column_stack([inits, h_init])
        ds, model, train_cell = self.generate_model(pdes, inits_)

        # Main training loop
        start_time = time.time()
        last_time = start_time
        for i in range(self.epochs):
            # Train a new model
            if np.mod(i, self.model_boundaries[self.nr_models]) == 0 and i > 0:
                self.nr_models += 1
                # Use the previous model
                t_init = (self.nr_models - 1) * self.t_final / self.nr_boundaries + 0 * t_init
                t_ms, x_ms, y_ms = to_tensor((np.expand_dims(t_init, axis=1), np.expand_dims(x_init, axis=1),
                                              np.expand_dims(y_init, axis=1)), dtype=self.data_type)
                h_init = model(t_ms, x_ms, y_ms)[:, 0].asnumpy()
                inits_ = np.column_stack([t_init, x_init, y_init, h_init])
                ckpt_path = f"{self.save_ckpt_path}/Optim_swe_{(self.nr_models - 1)}_{self.amp_level}.ckpt"
                ms.save_checkpoint(model, ckpt_path)
                ds, model, train_cell = self.generate_model(pdes, inits_)
                if self.load_ckpt:
                    print_log("Load weights from trained model...")
                    ckpt_path = f"{self.save_ckpt_path}/Optim_swe_{self.nr_models}_{self.amp_level}.ckpt"
                    ms.load_checkpoint(ckpt_path, model)
                else:
                    print_log("Copy over weights from previous model...")
                    ms.load_checkpoint(ckpt_path, model)

            pde_loss = Tensor(0)
            ic_loss = Tensor(0)
            for pdes_inits in ds.create_dict_iterator():
                pdes_, inits_ = pdes_inits['pde'], pdes_inits['init']
                # Train the network
                pde_loss, ic_loss = train_cell(pdes_, inits_)
                # Gradient step
                epoch_loss[i] += pde_loss + ic_loss
                nr_batches += 1
            # Get total epoch loss
            epoch_loss[i] /= nr_batches
            nr_batches = 0

            this_time = time.time()
            interval = this_time - last_time
            total_time = this_time - start_time
            last_time = this_time
            print_log("step: {}, PDE_loss: {:8.8f}, IC_loss: {:8.8f}, interval: {:8.8f}s, total: {:8.8f}s"
                      .format(i, pde_loss.asnumpy(), ic_loss.asnumpy(), interval, total_time))

        # Add the last model
        fname = f"{self.save_ckpt_path}/Optim_swe_{self.nr_models}_{self.amp_level}.ckpt"
        ms.save_checkpoint(model, fname)

        return epoch_loss

    def evaluate(self, pdes, inits):
        """evaluate"""
        # initial value points
        t_init, x_init, y_init = inits[:, 0], inits[:, 1], inits[:, 2]
        h_init = self.h0(x_init, y_init)
        inits_ = np.column_stack([t_init, x_init, y_init, h_init])
        ds, model, train_cell = self.generate_model(pdes, inits_)
        print_log("Load weights from trained model...")
        ckpt_path = f"{self.load_ckpt_path}"
        ms.load_checkpoint(ckpt_path, model)
        loss = 0
        for pdes_inits in ds.create_dict_iterator():
            pdes_, inits_ = pdes_inits['pde'], pdes_inits['init']
            pde_loss, ic_loss = train_cell.train_cell.network(pdes_, inits_)
            loss += pde_loss + ic_loss
        loss /= len(ds)
        return loss

    def predict(self, pdes, t, x, y):
        """predict"""
        num_pde = len(pdes)
        total_steps = len(t)

        steps_per_model = total_steps // self.nr_models
        h = np.zeros((total_steps, 1))

        nr_boundaries = len(self.model_boundaries) - 1

        for i in range(self.nr_models):
            idx_ds = (i + 1) * num_pde // nr_boundaries
            idx_ds_ = i * num_pde // nr_boundaries
            pts = pdes[idx_ds_:idx_ds]
            t_min, t_max = pts[:, 0].min(), pts[:, 0].max()
            model = Model(t_min, t_max, self.layers)
            if self.load_ckpt_path.endswith(".ckpt"):
                ms.load_checkpoint(self.load_ckpt_path, model)
            else:
                ckpt_path = f"{self.load_ckpt_path}/Optim_swe_{i + 1}_{self.amp_level}.ckpt"
                ms.load_checkpoint(ckpt_path, model)
            model.to_float(self.data_type)
            tt = ms.Tensor(t[i * steps_per_model:(i + 1) * steps_per_model, :], self.data_type)
            xx = ms.Tensor(x[i * steps_per_model:(i + 1) * steps_per_model, :], self.data_type)
            yy = ms.Tensor(y[i * steps_per_model:(i + 1) * steps_per_model, :], self.data_type)

            hh = model(tt, xx, yy)
            h[i * steps_per_model:(i + 1) * steps_per_model, :] = hh.asnumpy()

        return h

    def generate_model(self, pdes, inits_):
        """generate_model"""
        num_pde = len(pdes)
        num_iv = len(inits_)

        idx_ds = self.nr_models * num_pde // self.nr_boundaries
        idx_ds_ = (self.nr_models - 1) * num_pde // self.nr_boundaries
        pts = pdes[idx_ds_:idx_ds]
        t_min, t_max = pts[:, 0].min(), pts[:, 0].max()

        model = Model(t_min, t_max, self.layers)
        loss_net = Net(model, self.alpha_ms, self.u00_ms, self.u_ms)
        opt = nn.Adam(loss_net.trainable_params(), self.lr)
        train_cell = TrainCellWithCallBack(loss_net, opt, loss_interval=0, amp_level=self.amp_level)

        ds_init = dataset.NumpySlicesDataset(data=inits_, column_names=["init"], shuffle=False)
        ds_init = ds_init.map(operations=transforms.TypeCast(ms.float32), input_columns="init")
        bs_inits = num_iv // 10
        ds_init = ds_init.shuffle(num_iv).batch(bs_inits)
        ds_pde = dataset.NumpySlicesDataset(data=pdes[idx_ds_:idx_ds], column_names=["pde"], shuffle=False)
        ds_pde = ds_pde.map(operations=transforms.TypeCast(ms.float32), input_columns="pde")
        bs_pdes = num_pde // 10
        ds_pde = ds_pde.shuffle(idx_ds - idx_ds_).batch(bs_pdes)
        ds = ds_pde.zip(ds_init)

        print_log("\nBuild model no. {} for initial time {: 4.2f}, running over interval [{: 4.2f},{: 4.2f}] "
                  "with lr={}.\n".format(self.nr_models, inits_[0, 0], t_min, t_max, self.lr))

        return ds, model, train_cell
