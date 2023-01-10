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
The GROVER eval function.
"""
import os
import time
import datetime
import mindspore as ms
from mindspore import Tensor
from mindspore.common import set_seed
from mindspore.communication.management import init
from src.model.models import GROVEREmbedding, GroverFinetuneTask
from src.data.dataset import create_grover_dataset
from src.model_utils.config import config
from src.util.logger import get_logger
from src.util.utils import GroverMetric
from src.model_utils.device_adapter import get_rank_id, get_device_num, get_device_id
from src.model_utils.moxing_adapter import obs_url_to_env, download_from_qizhi, \
                                           download_from_qizhi_multi, upload_to_qizhi

set_seed(1)


def context_init(args):
    """
    Init context.
    """
    device_id = get_device_id()
    ms.set_context(mode=ms.GRAPH_MODE, device_target=args.device_target, save_graphs=False, device_id=device_id)

    ms.reset_auto_parallel_context()
    if args.run_distribute:
        init()
        args.device_num = get_device_num()
        args.rank = get_rank_id()
        parallel_mode = ms.ParallelMode.DATA_PARALLEL

    else:
        args.device_num = 1
        args.rank = 0
        parallel_mode = ms.ParallelMode.STAND_ALONE

    ms.set_auto_parallel_context(device_num=args.device_num,
                                 parallel_mode=parallel_mode,
                                 gradients_mean=True)


def load_parameters(args, network, file_name):
    """
    Load parameters for evaluating.
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


def run_eval():
    """
    Eval function.
    """
    # context
    context_init(config)
    # logger
    config.outputs_dir = os.path.join(config.eval_dir, config.data_file_eval,
                                      datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    config.logger = get_logger(config.outputs_dir, config.rank)
    config.logger.save_args(config)

    # dataset
    smiles_path = os.path.join(config.data_path_eval, config.data_file_eval + "_val.csv")
    feature_path = os.path.join(config.data_path_eval, config.data_file_eval + "_val.npz")
    config.scaler_path = os.path.join(config.save_dir, config.data_file_eval + "_scaler")

    dataset, labels_scaler = create_grover_dataset(config, smiles_path=smiles_path, feature_path=feature_path,
                                                   is_training=False)
    data_loader = dataset.create_dict_iterator(output_numpy=True)

    # network
    config.is_training = False
    grover_model = GROVEREmbedding(config)
    network = GroverFinetuneTask(config, grover_model, is_training=config.is_training)
    load_parameters(config, network, config.pretrained)
    network.set_train(False)

    metrics = GroverMetric(config, labels_scaler=labels_scaler, save_path=config.outputs_dir)
    start_time = time.time()
    for _, data in enumerate(data_loader):
        smiles = data["smiles"]
        features_batch = Tensor.from_numpy(data["features"])
        targets = Tensor.from_numpy(data["labels"])

        f_atoms = Tensor.from_numpy(data["f_atoms"])
        f_bonds = Tensor.from_numpy(data["f_bonds"])
        a2b = Tensor.from_numpy(data["a2b"])
        b2a = Tensor.from_numpy(data["b2a"])
        b2revb = Tensor.from_numpy(data["b2revb"])
        a2a = Tensor.from_numpy(data["a2a"])
        a_scope = data["a_scope"].tolist()
        b_scope = data["b_scope"].tolist()
        scope = (a_scope, b_scope)
        input_graph = (f_atoms, f_bonds, a2b, b2a, b2revb, a2a)
        preds = network(input_graph, scope, features_batch, targets)
        metrics.update(preds, targets, smiles)

    result = metrics.eval()
    cost_time = time.time() - start_time
    config.logger.info('eval result {}: {:.4f}'.format(config.metrics, result))
    config.logger.info('testing cost time %.2f h', cost_time / 3600.)


if __name__ == '__main__':
    config.parser_name = "eval"
    if config.enable_modelarts:
        if not os.path.exists(config.data_dir):
            os.makedirs(config.data_dir)
        if not os.path.exists(config.result_dir):
            os.makedirs(config.result_dir)
        # Initialize and copy data to training data
        obs_url_to_env(config.ckpt_url, config.ckpt_dir)
        if config.dataset_type == "regression":
            download_from_qizhi_multi(config.multi_data_url, config.data_dir)
            config.data_path_eval = os.path.join(config.data_dir, config.data_file_eval)
        else:
            download_from_qizhi(config.data_url, config.data_dir)
            config.data_path_eval = config.data_dir

        config.pretrained = config.ckpt_dir
        config.eval_dir = config.result_dir
        config.save_dir = config.data_dir

    run_eval()
    if config.enable_modelarts:
        upload_to_qizhi(config.result_dir, config.result_url)
