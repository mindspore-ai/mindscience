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
"""graphcast utils"""
import os

import matplotlib.pyplot as plt
import numpy as np
import mindspore.common.dtype as mstype
import mindspore.communication.management as D
from mindspore.communication import init
from mindspore import context, Tensor, ops, nn
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from mindearth.cell import GraphCastNet
from mindearth.data import FEATURE_DICT, SIZE_DICT
from mindearth.utils import make_dir, create_logger, plt_metrics
from .precip import PrecipNet


class OutputTo16(nn.Cell):
    "Wrap cell for amp. Cast network output back to float16"

    def __init__(self, op):
        super(OutputTo16, self).__init__(auto_prefix=False)
        self._op = op

    def construct(self, *x):
        return ops.functional.cast(self._op(*x), mstype.float16)


#pylint: disable=W0212
def amp_convert(network, black_list=None):
    """Do keep cell fp32."""
    network.to_float(mstype.float16)
    if black_list is not None:
        cells = network.name_cells()
        change = False
        for name in cells:
            subcell = cells[name]
            if subcell == network:
                continue
            elif isinstance(subcell, black_list):
                network._cells[name] = OutputTo16(subcell.to_float(mstype.float32))
                change = True
            else:
                amp_convert(subcell, black_list)
        if isinstance(network, nn.SequentialCell) and change:
            network.cell_list = list(network.cells())


def init_data_parallel(use_ascend):
    """Init data parallel for model running"""
    if use_ascend:
        init()
        device_num = D.get_group_size()
        os.environ['HCCL_CONNECT_TIMEOUT'] = "7200"
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=context.ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                          device_num=device_num, parameter_broadcast=False)
    else:
        init("nccl")
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=context.ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                          parameter_broadcast=True)


def load_dir_data(dirs, file_name, dtype=mstype.int32):
    """Load data"""
    path = os.path.join(dirs, file_name)
    return Tensor(np.load(path), dtype)


def init_model(config, run_mode="train"):
    """Init model"""
    data_params = config.get("data")
    model_params = config.get("model")
    train_params = config.get("train")
    summary_params = config.get("summary")
    train_params['load_ckpt'] = run_mode == "test"
    model_params['recompute'] = data_params.get("grid_resolution") < 1.0
    data_params['h_size'], data_params['w_size'] = SIZE_DICT[data_params.get("grid_resolution")]
    summary_params["summary_dir"] = get_model_summary_dir(config)
    make_dir(os.path.join(summary_params.get("summary_dir"), "image"))
    grid_mesh_info = GridMeshInfo(data_params)
    model = GraphCastNet(vg_in_channels=data_params.get('feature_dims') * data_params.get('t_in'),
                         vg_out_channels=data_params.get('feature_dims'),
                         vm_in_channels=model_params.get('vm_in_channels'),
                         em_in_channels=model_params.get('em_in_channels'),
                         eg2m_in_channels=model_params.get('eg2m_in_channels'),
                         em2g_in_channels=model_params.get('em2g_in_channels'),
                         latent_dims=model_params.get('latent_dims'),
                         processing_steps=model_params.get('processing_steps'),
                         g2m_src_idx=grid_mesh_info.g2m_src_idx,
                         g2m_dst_idx=grid_mesh_info.g2m_dst_idx,
                         m2m_src_idx=grid_mesh_info.m2m_src_idx,
                         m2m_dst_idx=grid_mesh_info.m2m_dst_idx,
                         m2g_src_idx=grid_mesh_info.m2g_src_idx,
                         m2g_dst_idx=grid_mesh_info.m2g_dst_idx,
                         mesh_node_feats=grid_mesh_info.mesh_node_feats,
                         mesh_edge_feats=grid_mesh_info.mesh_edge_feats,
                         g2m_edge_feats=grid_mesh_info.g2m_edge_feats,
                         m2g_edge_feats=grid_mesh_info.m2g_edge_feats,
                         per_variable_level_mean=grid_mesh_info.sj_mean,
                         per_variable_level_std=grid_mesh_info.sj_std,
                         recompute=model_params.get('recompute', False))
    if train_params.get('load_ckpt'):
        params = load_checkpoint(summary_params.get("ckpt_path"))
        load_param_into_net(model, params)
    return model


def init_tp_model(config, run_mode="train"):
    """Init precipitation model"""
    gc_no_grad = init_model(config)
    summary_params = config.get("summary")
    train_params = config.get("train")
    if run_mode == "train":
        params = load_checkpoint(summary_params.get("backbone_ckpt_path"))
        load_param_into_net(gc_no_grad, params)
    gc_grad = init_model(config)
    model = PrecipNet(gc_no_grad, gc_grad, config.get("data"))
    train_params['load_ckpt'] = run_mode == "test"
    if train_params.get("load_ckpt", False):
        params = load_checkpoint(config['summary']["ckpt_path"])
        load_param_into_net(model, params)
    if train_params.get("mixed_precision", True):
        fp32_black_list = (nn.GELU, nn.Softmax, nn.BatchNorm2d, nn.LayerNorm)
        amp_convert(model, fp32_black_list)
    return model


def get_model_summary_dir(config):
    """Get model summary directory"""
    model_params = config.get('model')
    model_name = model_params.get('name')
    summary_dir = model_name
    optimizer_params = config.get('optimizer')
    train_params = config.get('train')
    for k in model_params.keys():
        if k == 'name':
            continue
        summary_dir += '_{}_{}'.format(k, str(model_params[k]))
    summary_dir += '_{}'.format(optimizer_params.get('name'))
    summary_dir += '_{}'.format(train_params.get('name'))
    return summary_dir


def get_logger(config):
    """Get logger for saving log"""
    summary_params = config.get('summary')
    logger = create_logger(path=os.path.join(summary_params.get("summary_dir"), "results.log"))
    for key in config:
        logger.info(config[key])
    return logger


def get_coe(config):
    """Get coe"""
    data_params = config.get("data")
    w_size = data_params.get("w_size")
    coe_dir = os.path.join(data_params.get("root_dir"), "coe")
    sj_std = load_dir_data(coe_dir, 'sj_std.npy', mstype.float32)
    wj = load_dir_data(coe_dir, 'wj.npy', mstype.float32)
    ori_ai = np.load(os.path.join(coe_dir, "ai_norm.npy"))
    ori_ai = np.repeat(ori_ai, w_size, axis=-1)
    ori_ai = ori_ai.reshape(1, -1)
    ori_ai = np.repeat(ori_ai, data_params.get('batch_size'), axis=0)
    ori_ai = ori_ai.reshape(-1, 1)
    ai = Tensor(ori_ai, mstype.float32)
    return sj_std, wj, ai


def get_param_dict(config, current_step, steps_per_epoch, rollout_ckpt_pth=None):
    """Get param dict when load checkpoint"""
    if current_step == 1 and rollout_ckpt_pth:
        params_dict = load_checkpoint(rollout_ckpt_pth)
    else:
        optimizer_params = config.get('optimizer')
        model_params = config.get('model')
        summary_params = config.get("summary")
        train_params = config.get("train")
        if current_step > 1:
            optimizer_params['epochs'] = optimizer_params.get('finetune_epochs')
        if train_params.get('distribute'):
            ckpt_name = model_params.get('name') + '-' + 'device' + os.getenv("DEVICE_ID") + '-' + str(
                optimizer_params.get('epochs')) + '_' + str(steps_per_epoch) + '.ckpt'
        else:
            ckpt_name = model_params.get('name') + '-' + str(optimizer_params.get('epochs')) + '_' + str(
                steps_per_epoch) + '.ckpt'
        ckpt_path = os.path.join(summary_params.get('summary_dir'), 'ckpt', f'step_{current_step}', ckpt_name)
        params_dict = load_checkpoint(ckpt_path)
    return params_dict, ckpt_path


def _get_absolute_idx(feature_tuple, pressure_level_num):
    """Get absolute index in metrics"""
    return feature_tuple[1] * pressure_level_num + feature_tuple[0]


def plt_key_info(key_info, config, epochs=1, metrics_type='RMSE', loc='upper right'):
    """ Visualize the rmse or acc results, metrics_type is 'Acc' or 'RMSE' """
    data_params = config.get('data')
    summary_params = config.get('summary')
    pred_lead_time = data_params.get('pred_lead_time', 6)
    x = range(pred_lead_time, data_params.get('t_out_valid', 20) * pred_lead_time + 1, pred_lead_time)
    z500_idx = _get_absolute_idx(FEATURE_DICT.get("Z500"), data_params.get('pressure_level_num'))
    t2m_idx = _get_absolute_idx(FEATURE_DICT.get("T2M"), data_params.get('pressure_level_num'))
    t850_idx = _get_absolute_idx(FEATURE_DICT.get("T850"), data_params.get('pressure_level_num'))
    u10_idx = _get_absolute_idx(FEATURE_DICT.get("U10"), data_params.get('pressure_level_num'))
    xaxis_interval = plt.MultipleLocator(24)

    plt.figure(figsize=(14, 7))
    ax1 = plt.subplot(2, 2, 1)
    plt_metrics(x, key_info[z500_idx, :], "Z500", "Z500", ylabel=metrics_type, loc=loc)
    ax1.xaxis.set_major_locator(xaxis_interval)
    ax2 = plt.subplot(2, 2, 2)
    plt_metrics(x, key_info[t2m_idx, :], "T2M", "T2M", ylabel=metrics_type, loc=loc)
    ax2.xaxis.set_major_locator(xaxis_interval)
    ax3 = plt.subplot(2, 2, 3)
    plt_metrics(x, key_info[t850_idx, :], "T850", "T850", ylabel=metrics_type, loc=loc)
    ax3.xaxis.set_major_locator(xaxis_interval)
    ax4 = plt.subplot(2, 2, 4)
    plt_metrics(x, key_info[u10_idx, :], "U10", "U10", ylabel=metrics_type, loc=loc)
    ax4.xaxis.set_major_locator(xaxis_interval)
    plt.subplots_adjust(wspace=0.25, hspace=0.6)
    plt.savefig(f"{summary_params.get('summary_dir')}/image/Eval_{metrics_type}_epoch{epochs}.png", bbox_inches="tight")


def unlog_trans(x, eps=1e-5):
    """Inverse transformation of log(TP / epsilon + 1)"""
    return eps * (ops.exp(x) - 1)


class GridMeshInfo:
    """Init grid mesh information"""
    def __init__(self, data_params):
        level = data_params.get("mesh_level")
        resolution = data_params.get('grid_resolution')
        self.g2m_dst_idx, self.g2m_src_idx, self.m2g_dst_idx, \
            self.m2g_src_idx, self.m2m_dst_idx, self.m2m_src_idx = self._get_idx(level, data_params, resolution)
        self.g2m_edge_feats, self.m2g_edge_feats, \
            self.mesh_edge_feats, self.mesh_node_feats = self._get_feats(level, data_params, resolution)
        self.sj_mean, self.sj_std = self._get_sj_info(data_params)

    @staticmethod
    def _get_sj_info(data_params):
        coe_dir = os.path.join(data_params.get("root_dir"), "coe")
        sj_mean = load_dir_data(coe_dir, 'sj_mean.npy', mstype.float32)
        sj_std = load_dir_data(coe_dir, 'sj_std.npy', mstype.float32)
        return sj_mean, sj_std

    @staticmethod
    def _get_idx(level, data_params, resolution):
        mesh_dir = os.path.join(data_params.get("root_dir"), f'geometry_level{level}_resolution{resolution}')
        g2m_dst_idx = load_dir_data(mesh_dir, f'g2m_receiver_level{level}_r{resolution}.npy')
        g2m_src_idx = load_dir_data(mesh_dir, f'g2m_sender_level{level}_r{resolution}.npy')
        m2m_dst_idx = load_dir_data(mesh_dir, f'mesh_edge_receiver_level0_{level}_r{resolution}.npy')
        m2m_src_idx = load_dir_data(mesh_dir, f'mesh_edge_sender_level0_{level}_r{resolution}.npy')
        m2g_dst_idx = load_dir_data(mesh_dir, f'm2g_receiver_level{level}_r{resolution}.npy')
        m2g_src_idx = load_dir_data(mesh_dir, f'm2g_sender_level{level}_r{resolution}.npy')
        return g2m_dst_idx, g2m_src_idx, m2g_dst_idx, m2g_src_idx, m2m_dst_idx, m2m_src_idx

    @staticmethod
    def _get_feats(level, data_params, resolution):
        """Get graph features"""
        mesh_dir = os.path.join(data_params.get("root_dir"), f'geometry_level{level}_resolution{resolution}')
        mesh_node_feats = load_dir_data(mesh_dir,
                                        f'mesh_node_features_level{level}_r{resolution}.npy',
                                        mstype.float32)
        mesh_edge_feats = load_dir_data(mesh_dir,
                                        f'mesh_edge_normalization_level0_{level}_r{resolution}.npy',
                                        mstype.float32)
        g2m_edge_feats = load_dir_data(mesh_dir,
                                       f'g2m_edge_normalization_level{level}_r{resolution}.npy',
                                       mstype.float32)
        m2g_edge_feats = load_dir_data(mesh_dir,
                                       f'm2g_edge_normalization_level{level}_r{resolution}.npy',
                                       mstype.float32)
        return g2m_edge_feats, m2g_edge_feats, mesh_edge_feats, mesh_node_feats
