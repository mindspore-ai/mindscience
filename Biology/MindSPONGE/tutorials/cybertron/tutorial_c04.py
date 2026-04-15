# Copyright 2021-2023 @ Shenzhen Bay Laboratory &
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
Cybertron tutorial 04: Read parameters and hyperparameters from checkpoint file
                       and to use test dataset with scale and shift.
"""

import sys
import numpy as np
from mindspore import nn
from mindspore import context
from mindspore import dataset as ds
from mindspore.train import Model
from mindspore.train import load_checkpoint

if __name__ == '__main__':

    sys.path.append('..')

    from mindsponge.data import load_hyperparam
    from cybertron import Cybertron
    from cybertron.train import MAE
    from cybertron.train import WithLabelEvalCell

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    ckpt_file = 'Tutorial_C03/Tutorial_C03_MolCT-best.ckpt'
    hyper_param = load_hyperparam(ckpt_file)

    net = Cybertron(hyper_param=hyper_param)
    load_checkpoint(ckpt_file, net)

    idx = [7]  # U0

    test_file = sys.path[0] + '/dataset_qm9_origin_testset_1024.npz'
    test_data = np.load(test_file)

    scale = test_data['scale'][idx]
    shift = test_data['shift'][idx]
    ref = test_data['type_ref'][:, idx]

    net.set_scaleshift(scale=scale, shift=shift, type_ref=ref, unit='kj/mol')
    net.print_info()

    ds_test = ds.NumpySlicesDataset(
        {'R': test_data['R'], 'Z': test_data['Z'], 'E': test_data['E'][:, idx]}, shuffle=False)
    ds_test = ds_test.batch(1024)
    ds_test = ds_test.repeat(1)

    eval_network = WithLabelEvalCell('RZE', net, nn.MAELoss())

    eval_mae = 'EvalMAE'
    atom_mae = 'AtomMAE'
    model = Model(net, eval_network=eval_network, metrics=
                  {eval_mae: MAE([1, 2]), atom_mae: MAE([1, 2, 3], averaged_by_atoms=True)})

    print('Test dataset:')
    eval_metrics = model.eval(ds_test, dataset_sink_mode=False)
    info = ''
    for k, v in eval_metrics.items():
        info += k
        info += ': '
        info += str(v)
        info += ', '
    print(info)

    outdir = 'Tutorial_C04'
    scaled_ckpt = outdir + '_' + net.model_name + '.ckpt'
    net.save_checkpoint(scaled_ckpt, outdir)

    net2 = Cybertron(hyper_param=load_hyperparam(outdir+'/'+scaled_ckpt))
    net2.print_info()
