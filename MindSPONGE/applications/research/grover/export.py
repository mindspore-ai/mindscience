# Copyright 2022 Huawei Technologies Co., Ltd
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
Export the network.
"""
import os
import datetime
import numpy as np
import mindspore as ms
from mindspore import context, Tensor
from mindspore.train.serialization import export
from src.model.models import GROVEREmbedding, GroverFinetuneTask, GroverExportTask
from src.model_utils.config import config
from src.data.dataset import create_grover_dataset
from src.util.logger import get_logger


def load_parameters(args, network, file_name):
    """
    Load parameters.
    """
    args.logger.info("grover pretrained network model: %s", file_name)
    param_dict = ms.load_checkpoint(file_name)
    param_dict_new = {}
    filter_key = {}
    for key, values in param_dict.items():
        if key.startswith('grover.') or key.startswith('mol'):
            if key in filter_key:
                continue
            param_dict_new[key] = values
            args.logger.info('in resume {}'.format(key))
        else:
            continue
    ms.load_param_into_net(network, param_dict_new)
    args.logger.info('load_model %s success', file_name)


def run_export():
    """
    Export the network.
    """
    devid = int(os.getenv('DEVICE_ID', '0'))
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, device_id=devid)

    # logger
    config.outputs_dir = os.path.join(config.eval_dir, config.data_file_eval,
                                      datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    config.logger = get_logger(config.outputs_dir, config.rank)
    config.logger.save_args(config)

    # dataset
    smiles_path = os.path.join(config.data_path_eval, config.data_file_eval + "_val.csv")
    feature_path = os.path.join(config.data_path_eval, config.data_file_eval + "_val.npz")
    config.scaler_path = os.path.join(config.save_dir, config.data_file_eval + "_scaler")
    if not os.path.exists(config.scaler_path):
        os.makedirs(config.scaler_path, exist_ok=True)

    dataset, _ = create_grover_dataset(config, smiles_path=smiles_path, feature_path=feature_path,
                                       is_training=False)
    data_loader = dataset.create_dict_iterator(output_numpy=True)

    f_atoms = None
    f_bonds = None
    a2b = None
    b2a = None
    b2revb = None
    a2a = None
    a_scope = None
    b_scope = None
    features_batch = None

    for step_idx, data in enumerate(data_loader):
        features_batch = Tensor.from_numpy(data["features"])
        f_atoms = Tensor.from_numpy(data["f_atoms"])
        f_bonds = Tensor.from_numpy(data["f_bonds"])
        a2b = Tensor.from_numpy(data["a2b"])
        b2a = Tensor.from_numpy(data["b2a"])
        b2revb = Tensor.from_numpy(data["b2revb"])
        a2a = Tensor.from_numpy(data["a2a"])

        a_scope = Tensor(data["a_scope"].tolist())
        b_scope = Tensor(data["b_scope"].tolist())

        if step_idx == 0:
            break

    config.is_training = False
    grover_model = GROVEREmbedding(config)
    network = GroverFinetuneTask(config, grover_model, is_training=config.is_training)
    load_parameters(config, network, config.pretrained)
    network = GroverExportTask(network)
    network.set_train(False)

    f_atoms = Tensor(np.zeros(f_atoms.shape, np.float32))
    f_bonds = Tensor(np.zeros(f_bonds.shape, np.float32))
    a2b = Tensor(np.zeros(a2b.shape, np.int32))
    b2a = Tensor(np.zeros(b2a.shape, np.int32))
    b2revb = Tensor(np.zeros(b2revb.shape, np.int32))
    a2a = Tensor(np.zeros(a2a.shape, np.int32))
    a_scope = Tensor(np.zeros(a_scope.shape, np.int64))
    b_scope = Tensor(np.zeros(b_scope.shape, np.int64))
    features_batch = Tensor(np.zeros(features_batch.shape, np.float32))

    export(network, f_atoms, f_bonds, a2b, b2a, b2revb, a2a, a_scope, b_scope, features_batch,
           file_name="GROVER" + config.data_file_eval, file_format=config.file_format)


if __name__ == "__main__":
    run_export()
