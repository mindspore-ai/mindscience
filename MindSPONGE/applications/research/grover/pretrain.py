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
The GROVER pretrain function.
"""
import os
import time
import datetime
import mindspore as ms
from mindspore import nn, mutable
from mindspore.common import set_seed
from mindspore.communication.management import init
from src.model.models import GROVEREmbedding, GroverPretrainTask, GroverPretrainLossBlock
from src.util.nn_utils import SelectIndex
from src.util.scheduler import get_lr
from src.data.dataset import create_pretrain_dataset
from src.util.utils import AverageMeter
from src.util.logger import get_logger
from src.model_utils.config import config
from src.model_utils.device_adapter import get_rank_id, get_device_num, get_device_id
from src.model_utils.moxing_adapter import download_from_qizhi, upload_to_qizhi

set_seed(1)


def context_init(args):
    """
    Init context.
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


def run_training():
    """
    Run training function.
    """
    # context
    context_init(config)
    # logger
    config.outputs_dir = os.path.join(config.save_dir,
                                      datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    config.logger = get_logger(config.outputs_dir, config.rank)
    config.logger.save_args(config)

    smiles_path = os.path.join(config.data_path_pretrain, config.data_file_pretrain + "_train.csv")
    feature_path = os.path.join(config.data_path_pretrain, config.data_file_pretrain + "_train.npz")

    dataset, size_tuple = create_pretrain_dataset(config, smiles_path=smiles_path,
                                                  feature_path=feature_path,
                                                  num_shards=config.device_num,
                                                  shard_id=config.rank)

    # Build the network.
    config.is_training = True
    grover_model = GROVEREmbedding(config)
    network = GroverPretrainTask(config, grover_model, atom_vocab_size=size_tuple[0], bond_vocab_size=size_tuple[1],
                                 fg_size=size_tuple[2])

    config.steps_per_epoch = dataset.get_dataset_size()
    lr = get_lr(config)
    opt = nn.Adam(network.trainable_params(), learning_rate=ms.Tensor(lr), weight_decay=config.weight_decay)
    if config.mixed:
        loss_scale_manager = ms.FixedLossScaleManager(config.loss_scale_value, drop_overflow_update=False)
        network = ms.build_train_network(network, optimizer=opt, loss_scale_manager=loss_scale_manager,
                                         level="O2", keep_batchnorm_fp32=False)

        for _, cell in network.cells_and_names():
            if isinstance(cell, (GroverPretrainLossBlock, nn.Softmax, nn.LayerNorm, SelectIndex)):
                cell.to_float(ms.float32)

    else:
        network = nn.TrainOneStepCell(network=network, optimizer=opt)
    network.set_train(True)

    data_loader = dataset.create_dict_iterator(output_numpy=False)

    loss_meter = AverageMeter("loss")
    t_end = time.time()

    for epoch_idx in range(config.epochs):
        for step_idx, data in enumerate(data_loader):
            a_scope = data["a_scope"].asnumpy().tolist()
            b_scope = data["b_scope"].asnumpy().tolist()
            scope = (a_scope, b_scope)
            input_graph = (data["f_atoms"], data["f_bonds"], data["a2b"], data["b2a"], data["b2revb"], data["a2a"])
            input_graph = mutable(input_graph)
            targets = (data["atom_vocab_label"], data["bond_vocab_label"], data["fgroup_label"])
            targets = mutable(targets)
            loss = network(input_graph, scope, targets)
            config.logger.info('step_id:{}, loss: {:.6f}'.format(step_idx, loss.asnumpy()))
            loss_meter.update(loss.asnumpy())

        time_used = time.time() - t_end
        per_step_time = time_used / config.steps_per_epoch
        config.logger.info('epoch[{}], loss: {:.6f}, per step time: {:.4f}s'.format(epoch_idx + 1,
                                                                                    loss_meter.avg, per_step_time))
        t_end = time.time()
        loss_meter.reset()

        if config.rank_save_ckpt_flag:
            ckpt_path = os.path.join(config.outputs_dir, 'ckpt_' + str(config.rank))
            if not os.path.exists(ckpt_path):
                os.makedirs(ckpt_path)
            ckpt_name = os.path.join(ckpt_path, "grover_{}.ckpt".format(epoch_idx + 1))
            ms.save_checkpoint(network, ckpt_name)


if __name__ == '__main__':
    config.parser_name = "pretrain"
    if config.enable_modelarts:
        if not os.path.exists(config.data_dir):
            os.makedirs(config.data_dir)
        if not os.path.exists(config.train_dir):
            os.makedirs(config.train_dir)
        # Initialize and copy data to training data
        download_from_qizhi(config.data_url, config.data_dir)
        config.save_dir = config.train_dir
        config.data_path_pretrain = config.data_dir
        config.atom_vocab_path = os.path.join(config.data_path_pretrain, "tryout_atom_vocab.pkl")
        config.bond_vocab_path = os.path.join(config.data_path_pretrain, "tryout_bond_vocab.pkl")

    run_training()
    if config.enable_modelarts:
        upload_to_qizhi(config.train_dir, config.train_url)
