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
"""dem utils"""
import os
import matplotlib.pyplot as plt

from mindspore.train import load_checkpoint, load_param_into_net
import mindspore.communication.management as D

from mindearth.cell import DEMNet
from mindearth.utils import make_dir


def init_model(config):
    r"""init model."""
    model_params = config["model"]
    model = DEMNet(in_channels=model_params["in_channels"],
                   out_channels=model_params["out_channels"],
                   kernel_size=model_params["kernel_size"],
                   scale=model_params["scale"])

    if config['train']['load_ckpt']:
        ckpt_path = os.path.join(config['summary']["summary_dir"], config['summary']["ckpt_path"])
        params = load_checkpoint(ckpt_path)
        load_param_into_net(model, params)
    return model


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


def update_config(opt, config):
    r"""updata configs."""
    make_dir(opt.output_dir)
    config['train']['distribute'] = opt.distribute
    config['train']['device_id'] = opt.device_id
    config['train']['amp_level'] = opt.amp_level
    config['train']['run_mode'] = opt.run_mode
    config['train']['load_ckpt'] = opt.load_ckpt
    if config['train']['run_mode'] == 'test':
        config['train']['load_ckpt'] = True

    config['data']['num_workers'] = opt.num_workers
    config['data']['epochs'] = opt.epochs

    config['summary']["valid_frequency"] = opt.valid_frequency
    summary_dir = get_model_summary_dir(config)
    config['summary']["summary_dir"] = os.path.join(opt.output_dir, summary_dir)
    make_dir(config['summary']["summary_dir"])
    config['summary']["ckpt_path"] = opt.ckpt_path


def init_data_parallel(use_ascend):
    r"""init data parallel."""
    if use_ascend:
        init()
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
        device_num = D.get_group_size()
        os.environ['HCCL_CONNECT_TIMEOUT'] = "1007200"
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=context.ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                          device_num=device_num, parameter_broadcast=False)
    else:
        init("nccl")
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=context.ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                          parameter_broadcast=True)

def plt_dem_data(image_array, title):
    r"""plot dem result data."""
    plt.imshow(X=image_array)
    plt.axis('off')
    plt.title(title, color='black', fontsize=8)
    cb = plt.colorbar(fraction=0.025)
    cb.ax.tick_params(labelsize=10)
