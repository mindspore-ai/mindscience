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
"""dgmr utils"""
import os
import numpy as np
import matplotlib.pyplot as plt

from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import mindspore.communication.management as D
from mindspore.communication import init

from mindearth.cell import DgmrGenerator, DgmrDiscriminator
from mindearth.utils import make_dir


def init_data_parallel(use_ascend):
    r"""init data parallel."""
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


def init_model(config):
    r"""init model."""
    model_params = config["model"]

    g_model = DgmrGenerator(
        forecast_steps=model_params["forecast_steps"],
        in_channels=model_params["in_channels"],
        out_channels=model_params["out_channels"],
        conv_type=model_params["conv_type"],
        num_samples=model_params["num_samples"],
        grid_lambda=model_params["grid_lambda"],
        latent_channels=model_params["latent_channels"],
        context_channels=model_params["context_channels"],
        generation_steps=model_params["generation_steps"]
    )
    d_model = DgmrDiscriminator(
        in_channels=model_params["in_channels"],
        num_spatial_frames=model_params["num_spatial_frames"],
        conv_type=model_params["conv_type"]
    )
    if config['train']['load_ckpt']:
        ckpt_path = os.path.join(config['summary']["summary_dir"], config['summary']["ckpt_path"])
        params = load_checkpoint(ckpt_path)
        load_param_into_net(g_model, params)
    return g_model, d_model


def get_model_summary_dir(config):
    r"""get model summary."""
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
    r"""updata configs."""
    make_dir(args.output_dir)
    config['train']['distribute'] = args.distribute
    config['train']['device_id'] = args.device_id
    config['train']['amp_level'] = args.amp_level
    config['train']['run_mode'] = args.run_mode
    config['train']['load_ckpt'] = args.load_ckpt
    if config['train']['run_mode'] == 'test':
        config['train']['load_ckpt'] = True

    config['data']['num_workers'] = args.num_workers

    config['summary']['eval_interval'] = args.eval_interval
    config['summary']['keep_checkpoint_max'] = args.keep_checkpoint_max
    summary_dir = get_model_summary_dir(config)
    config['summary']["summary_dir"] = os.path.join(args.output_dir, summary_dir)
    make_dir(config['summary']["summary_dir"])
    config['summary']["ckpt_path"] = args.ckpt_path


def plt_crps_max(crps_max, ax, index):
    """
    Visualize the crps_max in different scale.

    Args:
         crps_max (dict): the score of crps max
         ax (subplot): subplot instance
         index (int): the index
    """
    ax.plot(range(0, 90, 5), crps_max, color='blue', linewidth=2, linestyle="--")
    ax.tick_params(length=8, width=4, labelsize=10)
    ax.set_ylim(0, 0.4)
    ax.tick_params(labelsize=20)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('grey')
    ax.spines['left'].set_linewidth(5)
    ax.spines['bottom'].set_color('grey')
    ax.spines['bottom'].set_linewidth(5)
    ax.set_xlabel('Prediction interval [min]', fontsize=18)
    ax.set_title(f"Pooling Scale [km] = {4 ** (index - 1)}", fontsize=18)
    ax.grid()


def plt_radar_data(x, y):
    """
    Visualize the forecast results in T+30 , T+60, T+90 min.

    Args:
        x (Tensor): The groundtruth of precipitation in 90 min.
        y (int): The prediction of precipitation in 90 min.
    """
    fig_num = 3
    fig = plt.figure(figsize=(20, 8))
    tget = np.expand_dims(y.asnumpy().squeeze(0), axis=-1)
    for i in range(1, fig_num + 1):
        ax = fig.add_subplot(2, fig_num, i)
        ax.imshow(tget[5 + (i - 1) * 6, ..., 0], vmin=0, vmax=10, cmap="jet")
    pred = x.asnumpy().squeeze(0)
    for i in range(fig_num + 1, 2 * fig_num + 1):
        ax = fig.add_subplot(2, fig_num, i)
        ax.imshow(pred[4 + (i - fig_num - 1) * 6, ...], vmin=0, vmax=10, cmap="jet")
    plt.savefig(f'pred_result.png')
    plt.show()
