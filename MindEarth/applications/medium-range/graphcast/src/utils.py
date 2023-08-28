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
from mindspore import context, Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from mindearth.cell import GraphCastNet
from mindearth.data import FEATURE_DICT, SIZE_DICT
from mindearth.utils import make_dir, create_logger, plt_metrics


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


def init_model(config):
    """Init model"""
    data_params = config["data"]
    model_params = config["model"]
    grid_mesh_info = GridMeshInfo(data_params)
    model = GraphCastNet(vg_in_channels=data_params['feature_dims'] * data_params['t_in'],
                         vg_out_channels=data_params['feature_dims'],
                         vm_in_channels=data_params['vm_in_channels'],
                         em_in_channels=data_params['em_in_channels'],
                         eg2m_in_channels=data_params['eg2m_in_channels'],
                         em2g_in_channels=data_params['em2g_in_channels'],
                         latent_dims=model_params['latent_dims'],
                         processing_steps=model_params['processing_steps'],
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
                         recompute=model_params['recompute'])
    if config['train']['load_ckpt']:
        params = load_checkpoint(config['summary']["ckpt_path"])
        load_param_into_net(model, params)
    return model


def get_model_summary_dir(config):
    """Get model summary directory"""
    model_name = config['model']['name']
    summary_dir = model_name
    for k in config['model'].keys():
        if k == 'name':
            continue
        summary_dir += '_{}_{}'.format(k, str(config['model'][k]))
    summary_dir += '_{}'.format(config['optimizer']['name'])
    summary_dir += '_{}'.format(config['train']['name'])
    return summary_dir


def update_config(args, config):
    """Update config by user specified args"""
    config['train']['distribute'] = args.distribute
    config['train']['device_target'] = args.device_target
    config['train']['device_id'] = args.device_id
    config['train']['amp_level'] = args.amp_level
    config['train']['run_mode'] = args.run_mode
    config['train']['load_ckpt'] = args.load_ckpt
    if config['train']['run_mode'] == 'test':
        config['train']['load_ckpt'] = True

    config['model']['data_sink'] = args.data_sink
    config['model']['latent_dims'] = args.latent_dims
    config['model']['processing_steps'] = args.processing_steps
    config['model']['recompute'] = args.grid_resolution < 1.0

    config['data']['num_workers'] = args.num_workers
    config['data']['mesh_level'] = args.mesh_level
    config['data']['grid_resolution'] = args.grid_resolution
    config['data']['h_size'], config['data']['w_size'] = SIZE_DICT[args.grid_resolution]

    config['optimizer']['epochs'] = args.epochs
    config['optimizer']['finetune_epochs'] = args.finetune_epochs
    config['optimizer']['initial_lr'] = args.initial_lr

    config['summary']["eval_interval"] = args.eval_interval
    summary_dir = get_model_summary_dir(config)
    config['summary']["summary_dir"] = os.path.join(args.output_dir, summary_dir)
    make_dir(os.path.join(config['summary']["summary_dir"], "image"))
    config['summary']["ckpt_path"] = args.ckpt_path


def get_logger(config):
    """Get logger for saving log"""
    logger = create_logger(path=os.path.join(config['summary']["summary_dir"], "results.log"))
    for key in config:
        logger.info(config[key])
    return logger


def get_coe(config):
    """Get coe"""
    data_params = config["data"]
    w_size = data_params["w_size"]
    coe_dir = os.path.join(data_params["root_dir"], "coe")
    sj_std = load_dir_data(coe_dir, 'sj_std.npy', mstype.float32)
    wj = load_dir_data(coe_dir, 'wj.npy', mstype.float32)
    ai = load_dir_data(coe_dir, 'ai_norm.npy', mstype.float32).repeat(
        w_size, axis=-1).reshape(
            (1, -1)).repeat(data_params['batch_size'], axis=0).reshape(-1, 1)
    return sj_std, wj, ai


def get_param_dict(config, current_step, steps_per_epoch, rollout_ckpt_pth=None):
    """Get param dict when load checkpoint"""
    if current_step == 1 and rollout_ckpt_pth:
        params_dict = load_checkpoint(rollout_ckpt_pth)
    else:
        if current_step > 1:
            config['optimizer']['epochs'] = config['optimizer']['finetune_epochs']
        if config['train']['distribute']:
            ckpt_name = config['model']['name'] + '-' + 'device' + os.getenv("DEVICE_ID") + '-' + str(
                config['optimizer']['epochs']) + '_' + str(steps_per_epoch) + '.ckpt'
        else:
            ckpt_name = config['model']['name'] + '-' + str(config['optimizer']['epochs']) + '_' + str(
                steps_per_epoch) + '.ckpt'
        ckpt_path = os.path.join(config['summary']['summary_dir'], 'ckpt', f'step_{current_step}', ckpt_name)
        params_dict = load_checkpoint(ckpt_path)
    return params_dict, ckpt_path


def _get_absolute_idx(feature_tuple, pressure_level_num):
    """Get absolute index in metrics"""
    return feature_tuple[1] * pressure_level_num + feature_tuple[0]


def plt_key_info(key_info, config, epochs=1, metrics_type='RMSE', loc='upper right'):
    """ Visualize the rmse or acc results, metrics_type is 'Acc' or 'RMSE' """
    pred_lead_time = config['data'].get('pred_lead_time', 6)
    x = range(pred_lead_time, config['data'].get('t_out_valid', 20) * pred_lead_time + 1, pred_lead_time)
    z500_idx = _get_absolute_idx(FEATURE_DICT.get("Z500"), config['data']['pressure_level_num'])
    t2m_idx = _get_absolute_idx(FEATURE_DICT.get("T2M"), config['data']['pressure_level_num'])
    t850_idx = _get_absolute_idx(FEATURE_DICT.get("T850"), config['data']['pressure_level_num'])
    u10_idx = _get_absolute_idx(FEATURE_DICT.get("U10"), config['data']['pressure_level_num'])

    plt.figure(1, figsize=(14, 7))
    plt.tight_layout()
    plt.subplots(2, 2)
    plt.subplot(2, 2, 1)
    plt_metrics(x, key_info[z500_idx, :], metrics_type + " of Z500", "Z500", loc=loc)
    plt.subplot(2, 2, 2)
    plt_metrics(x, key_info[t2m_idx, :], metrics_type + " of T2M", "T2M", loc=loc)
    plt.subplot(2, 2, 3)
    plt_metrics(x, key_info[t850_idx, :], metrics_type + " of T850", "T850", loc=loc)
    plt.subplot(2, 2, 4)
    plt_metrics(x, key_info[u10_idx, :], metrics_type + " of U10", "U10", loc=loc)
    plt.subplots_adjust(wspace=0.25, hspace=0.6)
    plt.savefig(f"{config['summary']['summary_dir']}/image/Eval_{metrics_type}_epoch{epochs}.png", bbox_inches="tight")


class GridMeshInfo:
    """Init grid mesh information"""
    def __init__(self, data_params):
        level = data_params["mesh_level"]
        resolution = data_params['grid_resolution']
        self.g2m_dst_idx, self.g2m_src_idx, self.m2g_dst_idx, \
            self.m2g_src_idx, self.m2m_dst_idx, self.m2m_src_idx = self._get_idx(level, data_params, resolution)
        self.g2m_edge_feats, self.m2g_edge_feats, \
            self.mesh_edge_feats, self.mesh_node_feats = self._get_feats(level, data_params, resolution)
        self.sj_mean, self.sj_std = self._get_sj_info(data_params)

    @staticmethod
    def _get_sj_info(data_params):
        coe_dir = os.path.join(data_params["root_dir"], "coe")
        sj_mean = load_dir_data(coe_dir, 'sj_mean.npy', mstype.float32)
        sj_std = load_dir_data(coe_dir, 'sj_std.npy', mstype.float32)
        return sj_mean, sj_std

    @staticmethod
    def _get_idx(level, data_params, resolution):
        mesh_dir = os.path.join(data_params["root_dir"], f'geometry_level{level}_resolution{resolution}')
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
        mesh_dir = os.path.join(data_params["root_dir"], f'geometry_level{level}_resolution{resolution}')
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