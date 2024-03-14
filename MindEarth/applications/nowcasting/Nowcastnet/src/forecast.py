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
# ==============================================================================
"""NowcastNet predictor"""
import time
import os

import numpy as np
import mindspore as ms
from mindspore import ops, nn, Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from .dataset import RadarData, NowcastDataset
from .visual import plt_img
from .utils import make_grid, warp


def cal_tp(pred=None, target=None, th=None):
    return ops.where(ops.logical_and(pred >= th, target >= th), Tensor(np.ones(target.shape)),
                     Tensor(np.zeros(target.shape))).sum(axis=(-1, -2))


def cal_tn(pred=None, target=None, th=None):
    return ops.where(ops.logical_and(pred < th, target < th), Tensor(np.ones(target.shape)),
                     Tensor(np.zeros(target.shape))).sum(axis=(-1, -2))


def cal_fp(pred=None, target=None, th=None):
    return ops.where(ops.logical_and(pred >= th, target < th), Tensor(np.ones(target.shape)),
                     Tensor(np.zeros(target.shape))).sum(axis=(-1, -2))


def cal_fn(pred=None, target=None, th=None):
    return ops.where(ops.logical_and(pred < th, target >= th), Tensor(np.ones(target.shape)),
                     Tensor(np.zeros(target.shape))).sum(axis=(-1, -2))


def cal_csi(pred=None, target=None, tp=None, fp=None, fn=None, threshold=16):
    if tp is None and fp is None and fn is None:
        tp = cal_tp(pred=pred, target=target, th=threshold)
        fp = cal_fp(pred=pred, target=target, th=threshold)
        fn = cal_fn(pred=pred, target=target, th=threshold)
    tp_fp_fn = tp + fp + fn
    csi = ops.div(tp, ops.where(tp_fp_fn > 1e-5, tp_fp_fn, 1e-5 + Tensor(np.zeros(tp_fp_fn.shape))))
    return csi.mean(axis=0)


def cal_csin(pred=None, target=None, threshold=16):
    max_pool = nn.MaxPool2d(kernel_size=5, stride=2)
    pred = max_pool(ops.cast(pred, ms.float16))
    target = max_pool(ops.cast(target, ms.float16))
    csin = cal_csi(pred=pred, target=target, threshold=threshold)
    return csin


class GenerationPredictor(nn.Cell):
    """
    Perform the model inference in Nowcastnet generation module.
    """
    def __init__(self, config, g_model, logger):
        super(GenerationPredictor, self).__init__()
        self.data_params = config.get("data")
        self.summary_params = config.get("summary")
        self.train_params = config.get("train")
        self.model_params = config.get("model")
        self.time_interval = self.data_params.get("data_frequency", 10)
        self.noise_scale = self.data_params.get("noise_scale", 32)
        self.threshold = self.summary_params.get("csin_threshold", 16)
        self.batch_size = self.data_params.get("batch_size", 1)
        self.w_size = self.data_params.get("w_size", 512)
        self.h_size = self.data_params.get("h_size", 512)
        self.ngf = self.model_params.get("ngf", 32)
        self.generator = g_model
        self.logger = logger
        self.vis_save_path = os.path.join(self.summary_params.get("summary_dir"), "img")
        if not os.path.exists(self.vis_save_path):
            os.makedirs(self.vis_save_path)
        if self.train_params.get("load_ckpt", True):
            self.load_ckpt()

    def load_ckpt(self):
        params = load_checkpoint(self.summary_params.get("generate_ckpt_path"))
        load_param_into_net(self.generator, params)

    def print_csi_metrics(self, metrics, info_prefix='CSIN'):
        metrics_info = [info_prefix]
        time_interval = self.data_params.get('data_frequency', 10)
        for timestep in self.summary_params.get('key_info_timestep'):
            cur_info = f"T+{timestep} min: {metrics[timestep // time_interval - 1]}"
            metrics_info.append(cur_info)
        self.logger.info(" ".join(metrics_info))
        return metrics

    def eval(self, dataset):
        """eval"""
        self.generator.set_train(False)
        self.logger.info("================================Start Evaluation================================")
        self.logger.info(f"The length of data is: {dataset.get_dataset_size()}")
        steps = 1
        plt_idx = [x // self.time_interval - 1 for x in self.data_params.get("key_info_timestep", [10, 60, 120])]
        metrics_list = list()
        for data in dataset.create_dict_iterator():
            t1 = time.time()
            inp, evo_result, labels = data.get("inputs"), data.get("evo"), data.get("labels")
            noise = ms.tensor(ms.numpy.randn((self.batch_size,
                                              self.ngf,
                                              self.h_size // self.noise_scale,
                                              self.w_size // self.noise_scale)), inp.dtype)
            pred = self.generator(inp, evo_result, noise)
            metrics = cal_csin(pred=pred, target=labels, threshold=self.threshold)
            np_metrics = metrics.asnumpy()
            metrics_list.append(np_metrics)
            self.print_csi_metrics(np_metrics, info_prefix=f'CSI Neighborhood threshold {self.threshold}')
            if self.summary_params.get("visual", True):
                for j in range(self.batch_size):
                    plt_img(field=pred[j].asnumpy(),
                            label=labels[j].asnumpy(),
                            idx=plt_idx,
                            fig_name=os.path.join(self.vis_save_path, f"generation_{steps}_{j}.jpg"),
                            evo=evo_result[j].asnumpy() * 128,
                            plot_evo=True)
            step_cost = (time.time() - t1) * 1000
            self.logger.info("step {}, cost: {:.2f} ms".format(steps, step_cost))
            steps += 1
        self.logger.info("================================End Evaluation================================")
        return metrics_list


class EvolutionPredictor:
    """
    Perform the model inference in Nowcastnet evolution module.
    """
    def __init__(self, config, model, logger):
        self.model = model
        self.logger = logger
        self.config = config
        self.summary_params = config.get('summary')
        self.train_params = config.get("train")
        self.data_params = config.get('data')
        self.time_interval = self.data_params.get('data_frequency', 10)
        self.batch_size = self.data_params.get("batch_size", 1)
        self.t_in = self.data_params.get("t_in", 9)
        self.t_out = self.data_params.get("t_out", 20)
        self.threshold = self.summary_params.get("csin_threshold", 16)
        sample_tensor = np.zeros(
            (1, 1, self.data_params.get("h_size", 512), self.data_params.get("w_size", 512))).astype(np.float32)
        self.grid = Tensor(make_grid(sample_tensor), ms.float32)
        self.vis_save_path = os.path.join(self.summary_params.get("summary_dir", ""), "img")
        if not os.path.exists(self.vis_save_path):
            os.makedirs(self.vis_save_path)
        self.test_dataset = self.get_dataset()
        self.dataset_size = self.test_dataset.get_dataset_size()

    def print_csi_metrics(self, metrics, info_prefix="CSIN"):
        metrics_info = [info_prefix]
        time_interval = self.data_params.get('data_frequency', 10)
        for timestep in self.summary_params.get('key_info_timestep'):
            cur_info = f"T+{timestep} min: {metrics[timestep // time_interval - 1]}"
            metrics_info.append(cur_info)
        self.logger.info(" ".join(metrics_info))
        return metrics

    def get_dataset(self):
        test_dataset_generator = RadarData(self.data_params, run_mode='test', module_name='evolution')
        test_dataset = NowcastDataset(test_dataset_generator,
                                      module_name='evolution',
                                      distribute=self.train_params.get('distribute'),
                                      num_workers=self.data_params.get('num_workers'),
                                      shuffle=False
                                      )
        test_dataset = test_dataset.create_dataset(self.batch_size)
        return test_dataset

    def eval(self, dataset):
        """eval"""
        self.model.set_train(False)
        self.logger.info("================================Start Evaluation================================")
        self.logger.info(f"The length of data is: {dataset.get_dataset_size()}")
        steps = 1
        plt_idx = [x // self.time_interval - 1 for x in self.data_params.get("key_info_timestep", [10, 60, 120])]
        metrics_list = list()
        for data in dataset.create_dict_iterator():
            inp = data.get("inputs")
            pred = self.forecast(inp)
            labels = inp[:, self.t_in:]
            metrics = cal_csin(pred=pred, target=labels, threshold=self.threshold)
            np_metrics = metrics.asnumpy()
            metrics_list.append(np_metrics)
            self.print_csi_metrics(np_metrics, info_prefix=f'CSI Neighborhood threshold {self.threshold}')
            if self.summary_params.get("visual", True):
                for j in range(self.batch_size):
                    plt_img(field=pred[j].asnumpy(),
                            label=labels[j].asnumpy(),
                            idx=plt_idx,
                            fig_name=os.path.join(self.vis_save_path, f"evolution_{steps}_{j}.jpg"))
            steps += 1
        self.logger.info("================================End Evaluation================================")
        return metrics_list

    def forecast(self, inputs):
        """forecast prediction"""
        intensity, motion = self.model(inputs)
        batch, _, height, width = inputs.shape
        motion_ = motion.reshape(batch, self.t_out, 2, height, width)
        intensity_ = intensity.reshape(batch, self.t_out, 1, height, width)
        series = list()
        last_frames = inputs[:, (self.t_in - 1):self.t_in, :, :]
        grid = self.grid.tile((batch, 1, 1, 1))
        for i in range(self.t_out):
            last_frames = warp(last_frames, motion_[:, i], grid, mode="nearest", padding_mode="border")
            last_frames = last_frames + intensity_[:, i]
            series.append(last_frames)
        pred = ops.cat(series, axis=1)
        return pred
