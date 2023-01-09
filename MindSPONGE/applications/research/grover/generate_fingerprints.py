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
The fingerprint generation function.
"""
import os
import time
import datetime
import numpy as np
import mindspore as ms
from mindspore.common import set_seed
from mindspore.communication.management import init
from src.model.models import GROVEREmbedding, GroverFpGenerationTask
from src.data.dataset import create_grover_dataset
from src.util.logger import get_logger
from src.model_utils.config import config
from src.model_utils.device_adapter import get_rank_id, get_device_num, get_device_id
from src.model_utils.moxing_adapter import download_from_qizhi_multi, upload_to_qizhi

set_seed(1)


def load_convert_params(args, network):
    """
    Load pretrained model parameters for finetuning.
    """
    if args.resume_grover:
        param_dict = ms.load_checkpoint(args.resume_grover)
        param_dict_new = {}
        for key, values in param_dict.items():
            param_dict_new[key] = values
            args.logger.info('in resume {}'.format(key))
        ms.load_param_into_net(network, param_dict_new)


def context_init(args):
    """
    Init Context.
    """
    device_id = get_device_id()
    ms.set_context(mode=ms.GRAPH_MODE, device_target=args.device_target, save_graphs=False, device_id=device_id)

    if ms.get_context("device_target") == "Ascend":
        ms.set_context(max_device_memory="10GB")

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

    args.rank_save_ckpt_flag = 0
    if args.is_save_on_master:
        if args.rank == 0:
            args.rank_save_ckpt_flag = 1
    else:
        args.rank_save_ckpt_flag = 1


def generate_fingeprints():
    """
    Generate the molecular fingerprints.
    """

    # context
    context_init(config)
    # logger
    config.outputs_dir = os.path.join(config.save_dir, 'fingerprints', config.data_file_fp,
                                      datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    config.logger = get_logger(config.outputs_dir, config.rank)
    config.logger.save_args(config)

    # dataset
    smiles_path = os.path.join(config.data_path_fp, config.data_file_fp + "_val.csv")
    feature_path = os.path.join(config.data_path_fp, config.data_file_fp + "_val.npz")

    dataset, _ = create_grover_dataset(config, smiles_path=smiles_path, feature_path=feature_path,
                                       num_shards=config.device_num, shard_id=config.rank)
    data_loader = dataset.create_dict_iterator(output_numpy=False)

    # network
    config.is_training = False
    grover_model = GROVEREmbedding(config)
    load_convert_params(config, grover_model)
    network = GroverFpGenerationTask(config, grover_model)
    network.set_train(False)
    start_time = time.time()
    preds_list = []
    for _, data in enumerate(data_loader):
        features_batch = data["features"]
        a_scope = data["a_scope"].asnumpy().tolist()
        b_scope = data["b_scope"].asnumpy().tolist()
        scope = (a_scope, b_scope)
        input_graph = (data["f_atoms"], data["f_bonds"], data["a2b"], data["b2a"], data["b2revb"], data["a2a"])
        preds = network(input_graph, scope, features_batch)
        preds_list.append(preds.asnumpy())

    cost_time = time.time() - start_time
    fp_path = os.path.join(config.outputs_dir, 'fp.npz')
    np.savez_compressed(fp_path, fps=preds_list)
    config.logger.info('fp cost time %.2f h', cost_time / 3600.)


if __name__ == '__main__':
    config.parser_name = "fingeprints"
    if config.enable_modelarts:
        if not os.path.exists(config.data_dir):
            os.makedirs(config.data_dir)
        if not os.path.exists(config.train_dir):
            os.makedirs(config.train_dir)
        # Initialize and copy data to training data
        download_from_qizhi_multi(config.multi_data_url, config.data_dir)
        config.save_dir = config.train_dir
        config.resume_grover = os.path.join(config.data_dir, "resume_grover", "convert_grover_base.ckpt")
        config.data_path_fp = os.path.join(config.data_dir, config.data_file_fp)
    generate_fingeprints()
    if config.enable_modelarts:
        upload_to_qizhi(config.train_dir, config.train_url)
