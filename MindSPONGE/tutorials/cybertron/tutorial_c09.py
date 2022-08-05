# Copyright 2021-2022 @ Shenzhen Bay Laboratory &
#                       Peking University &
#                       Huawei Technologies Co., Ltd
#
# This code is a part of Cybertron package.
#
# The Cybertron is open-source software based on the AI-framework:
# MindSpore (https://www.mindspore.cn/)
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
"""
Cybertron tutorial 09: MolCT with ACT
"""

import sys
import time
import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore import Tensor
from mindspore import context
from mindspore import dataset as ds
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig

if __name__ == '__main__':

    sys.path.append('..')

    from cybertron import Cybertron
    from cybertron import MolCT
    from cybertron import AtomwiseReadout
    from cybertron.train import MAE, RMSE, MLoss, MSELoss
    from cybertron.train import WithForceLossCell, WithForceEvalCell
    from cybertron.train import TrainMonitor
    from cybertron.train import TransformerLR

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    sys_name = 'dataset_ethanol_normed_'

    train_file = sys_name + 'trainset_1024.npz'
    valid_file = sys_name + 'validset_128.npz'

    train_data = np.load(train_file)
    valid_data = np.load(valid_file)

    atom_types = Tensor(train_data['Z'], ms.int32)
    scale = train_data['scale']
    shift = train_data['shift']

    mod = MolCT(
        cutoff=1,
        n_interaction=3,
        dim_feature=128,
        n_heads=8,
        max_cycles=10,
        length_unit='nm',
    )

    readout = AtomwiseReadout(mod, dim_output=1)
    net = Cybertron(mod, readout=readout,
                    atom_types=atom_types, length_unit='nm')

    net.print_info()

    tot_params = 0
    for i, param in enumerate(net.trainable_params()):
        tot_params += param.size
        print(i, param.name, param.shape)
    print('Total parameters: ', tot_params)

    n_epoch = 8
    repeat_time = 1
    batch_size = 32

    ds_train = ds.NumpySlicesDataset(
        {'R': train_data['R'], 'F': train_data['F'], 'E': train_data['E']}, shuffle=True)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.repeat(repeat_time)

    ds_valid = ds.NumpySlicesDataset(
        {'R': valid_data['R'], 'F': valid_data['F'], 'E': valid_data['E']}, shuffle=False)
    ds_valid = ds_valid.batch(128)
    ds_valid = ds_valid.repeat(1)

    force_dis = train_data['avg_force_dis']
    loss_fn = MSELoss(ratio_energy=1, ratio_forces=100, force_dis=force_dis)
    loss_network = WithForceLossCell('RFE', net, loss_fn)
    eval_network = WithForceEvalCell(
        'RFE', net, loss_fn, scale=scale, shift=shift)

    lr = TransformerLR(learning_rate=1., warmup_steps=4000, dimension=128)
    optim = nn.Adam(params=net.trainable_params(), learning_rate=lr)

    outdir = 'Tutorial_C09'
    outname = outdir + '_' + net.model_name

    energy_mae = 'EnergyMAE'
    forces_mae = 'ForcesMAE'
    forces_rmse = 'ForcesRMSE'
    eval_loss = 'EvalLoss'
    model = Model(loss_network, eval_network=eval_network, optimizer=optim,
                  metrics={eval_loss: MLoss(), energy_mae: MAE([1, 2]), forces_mae: MAE([3, 4]),
                           forces_rmse: RMSE([3, 4], atom_aggregate='sum')})

    record_cb = TrainMonitor(model, outname, per_epoch=1, avg_steps=32,
                             directory=outdir, eval_dataset=ds_valid, best_ckpt_metrics=forces_rmse)

    config_ck = CheckpointConfig(
        save_checkpoint_steps=32, keep_checkpoint_max=64, append_info=[net.hyper_param])
    ckpoint_cb = ModelCheckpoint(
        prefix=outname, directory=outdir, config=config_ck)

    print("Start training ...")
    beg_time = time.time()
    model.train(n_epoch, ds_train, callbacks=[
                record_cb, ckpoint_cb], dataset_sink_mode=False)
    end_time = time.time()
    used_time = end_time - beg_time
    m, s = divmod(used_time, 60)
    h, m = divmod(m, 60)
    print("Training Fininshed!")
    print("Training Time: %02d:%02d:%02d" % (h, m, s))
