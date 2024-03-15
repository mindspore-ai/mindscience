# Copyright 2024 Huawei Technologies Co., Ltd
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
kernel
"""
import time
import os
import numpy as np
import mindspore as ms
import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore import ops
from mindspore.amp import DynamicLossScaler
from mindchemistry.graph.loss import L2LossMask

from .model import Net
from .data import AijData
from .parse_configs import BaseConfig, TrainConfig, EvalConfig
from .utils import LossRecord, process_targets, set_random_seed, RevertDecayLR
from .e3modules import E3TensorDecompNet

class DatasetInfo:
    """
    DatasetInfo class
    """

    def __init__(self, spinful, index_to_z, orbital_types):

        self.spinful = spinful

        if isinstance(index_to_z, list):
            self.index_to_z = np.array(index_to_z)
        elif isinstance(index_to_z, np.ndarray):
            self.index_to_z = index_to_z
        else:
            raise NotImplementedError
        self.orbital_types = orbital_types

        self.z_to_index = np.full((100,), -1, dtype=np.int64)
        self.z_to_index[self.index_to_z] = np.arange(len(index_to_z))

    def __eq__(self, __o) -> bool:
        """
        class equal write
        """
        if __o.__class__ != __class__:
            raise ValueError
        a = __o.spinful == self.spinful
        b = np.all(__o.index_to_z == self.index_to_z)
        c = __o.orbital_types == self.orbital_types
        return a * b * c

    @classmethod
    def from_dataset(cls, dataset: AijData):
        """
        DatasetInfo from dataset
        """
        return cls(dataset.info['spinful'], dataset.info['index_to_Z'], dataset.info['orbital_types'])

class NetOutInfo:
    """
    NetOutInfo class
    """

    def __init__(self, target_blocks, dataset_info: DatasetInfo):
        self.target_blocks = target_blocks
        self.dataset_info = dataset_info
        self.blocks, self.js, self.slices = process_targets(dataset_info.orbital_types, dataset_info.index_to_z,
                                                            target_blocks)


    def merge(self, other):
        """
        NetOutInfo class merge
        """
        self.target_blocks.extend(other.target_blocks)
        self.blocks.extend(other.blocks)
        self.js.extend(other.js)
        length = self.slices.pop()
        for i in other.slices:
            self.slices.append(i + length)


class Dataset:
    """
    Dataset class
    """
    def __init__(self, dataset):
        super(Dataset, self).__init__()
        self.dataset = dataset

    def __getitem__(self, index):
        dataset_batch = self.dataset[index]
        return tuple(dataset_batch)

    def __len__(self):
        return len(self.dataset)


class DeepHE3Kernel:
    """
    DeepHE3Kernel class
    """

    def __init__(self):

        # how to determine kernel mode:
        # train mode: self.train_config is not None and self.eval_config is None
        # eval mode: self.eval_config is not None
        self.train_config = None
        self.eval_config = None

        self.dataset = None
        self.dataset_info = None
        self.net = None
        self.net_out_info = None
        self.construct_kernel = None

        self.train_info = None

    @staticmethod
    def train_process(config, scheduler, optimizer, train_loader, val_loader, net, construct_kernel):
        """
        DeepHE3Kernel class train_process
        """
        loss_scaler = DynamicLossScaler(scale_value=8192, scale_factor=2, scale_window=50)
        l2_loss_mask = L2LossMask()

        def forward_train(data_x, data_edge_index, data_edge_attr, data_label, data_mask, data_mask_length,
                          batch_input_x, mask_dim1, mask_dim2, mask_dim3):
            edge_fea = net(data_x, data_edge_index, data_edge_attr, data_mask_length, batch_input_x, mask_dim1,
                           mask_dim2, mask_dim3)
            h_pred = construct_kernel(edge_fea)
            mse_loss = l2_loss_mask(h_pred, data_label, data_mask, data_mask_length)[0]
            mse_loss = loss_scaler.scale(mse_loss)
            return mse_loss, h_pred

        def forward_val(data_x, data_edge_index, data_edge_attr, data_label, data_mask, data_mask_length, batch_input_x,
                        mask_dim1, mask_dim2, mask_dim3):
            edge_fea = net(data_x, data_edge_index, data_edge_attr, data_mask_length, batch_input_x, mask_dim1,
                           mask_dim2, mask_dim3)
            h_pred = construct_kernel(edge_fea)
            mse_loss = l2_loss_mask(h_pred, data_label, data_mask, data_mask_length)[0]
            return mse_loss, h_pred

        backward = ms.value_and_grad(forward_train, None, weights=net.trainable_params(), has_aux=True)

        def train_step(data_x, data_edge_index, data_edge_attr, data_label, data_mask, data_mask_length, batch_input_x,
                       mask_dim1, mask_dim2, mask_dim3):
            (mse_loss, h_pred), grads = backward(data_x, data_edge_index, data_edge_attr, data_label, data_mask,
                                                 data_mask_length, batch_input_x, mask_dim1, mask_dim2, mask_dim3)
            ### loss scale
            mse_loss = loss_scaler.unscale(mse_loss)
            grads = loss_scaler.unscale(grads)
            optimizer(grads)

            return mse_loss, h_pred

        def eval_test_step(data_x, data_edge_index, data_edge_attr, data_label, data_mask, data_mask_length,
                           batch_input_x, mask_dim1, mask_dim2, mask_dim3):
            mse_loss, h_pred = forward_val(data_x, data_edge_index, data_edge_attr, data_label, data_mask,
                                           data_mask_length, batch_input_x, mask_dim1, mask_dim2, mask_dim3)
            return mse_loss, h_pred

        print('\n------- Begin training -------')

        best_loss = 10000
        train_begin_time = time.time()
        epoch = scheduler.next_epoch
        learning_rate = optimizer.learning_rate.value()
        batch_input_x = ops.zeros((64, 412))

        while epoch < config.num_epoch and learning_rate > config.min_lr:
            print("=================================epoch: " + str(epoch))
            train_losses = LossRecord()
            step = 0
            for batch in train_loader:
                starttime = time.time()
                train_mse_loss, _ = train_step(batch[0], batch[1], batch[2], batch[9], batch[10], batch[11],
                                               batch_input_x, batch[12], batch[13], batch[14])
                endtime = time.time()
                print("----------------------train epoch: " + str(epoch) + "-------step: " + str(step))
                print("training time", endtime - starttime)
                print("learning rate", optimizer.learning_rate.value())
                print("train mse loss", train_mse_loss)
                train_losses.update(train_mse_loss, batch[11][0])
                step = step + 1
            print("epoch: ", epoch)
            print("last train loss:", train_losses.last_val)
            print("average train loss:", train_losses.avg)
            val_losses = LossRecord()
            step = 0
            for batch in val_loader:
                starttime = time.time()
                val_mse_loss, _ = eval_test_step(batch[0], batch[1], batch[2], batch[9], batch[10], batch[11],
                                                 batch_input_x, batch[12], batch[13], batch[14])
                endtime = time.time()
                print("----------------------eval epoch: " + str(epoch) + "-------step: " + str(step))
                print("evaluating time", endtime - starttime)
                print("learning rate", optimizer.learning_rate.value())
                print("val mse loss", val_mse_loss)
                val_losses.update(val_mse_loss, batch[11][0])
                step = step + 1
            print("epoch: ", epoch)
            print("last eval loss:", val_losses.last_val)
            print("average eval loss:", val_losses.avg)

            if val_losses.avg < best_loss:
                best_loss = val_losses.avg

            scheduler.step(val_losses.avg)
            epoch = scheduler.next_epoch

            print(f'Train finished, cost {time.time() - train_begin_time:.2f}s.')
            print("best loss: ", best_loss)
        print('\nTraining finished.')
    def load_config(self, train_config_path=None, eval_config_path=None):
        """
        DeepHE3Kernel class load_config
        """
        if train_config_path is not None:
            self.train_config = TrainConfig(train_config_path)
        if eval_config_path is not None:
            self.eval_config = EvalConfig(eval_config_path)

    def preprocess(self, preprocess_config):
        """
        DeepHE3Kernel class preprocess
        """
        config = BaseConfig(preprocess_config)
        self.get_graph(config)

    def train(self, train_config):
        """
        DeepHE3Kernel class train process
        """
        ms.set_seed(1234)
        self.load_config(train_config_path=train_config)
        config = self.train_config

        # = record output =
        os.makedirs(config.save_dir)

        print('\n------- DeepH-E3 model training begins -------')
        set_random_seed(config.seed)
        dataset = self.get_graph(config)
        self.config_set_target()
        # set dataset mask
        dataset.set_mask(config.target_blocks)
        # = data loader =
        train_loader, val_loader = self.get_loader()
        # = Build net =
        net = self.build_model()
        print("finish load model")
        model_parameters = filter(lambda p: p.requires_grad, net.get_parameters())
        params = sum([np.prod(p.shape) for p in model_parameters])
        print("The model you built has %d parameters." % params)

        self.register_constructor()
        print(net)

        learning_rate = 0.003
        optimizer = nn.Adam(params=net.trainable_params(),
                            learning_rate=learning_rate,
                            beta1=config.adam_betas[0],
                            beta2=config.adam_betas[1])

        # = LR scheduler =
        scheduler = RevertDecayLR(net, optimizer, config.save_dir, config.revert_decay_patience,
                                  config.revert_decay_rate, config.scheduler_type, config.scheduler_params)

        print('Starting new training process')

        net = self.net
        config = self.train_config
        construct_kernel = self.construct_kernel

        self.train_process(config, scheduler, optimizer, train_loader, val_loader, net, construct_kernel)

    def get_graph(self, config: BaseConfig, inference=False):
        """
        DeepHE3Kernel get graph process
        """
        process_only = config.__class__ == BaseConfig
        # prepare graph data
        print('\nProcessing graph data...')
        dataset = AijData(raw_data_dir=config.processed_data_dir,
                          graph_dir=config.save_graph_dir,
                          target=config.target_data,
                          dataset_name=config.dataset_name,
                          multiprocessing=False,
                          radius=-1,
                          max_num_nbr=0,
                          edge_aij=True,
                          inference=inference,
                          only_ij=False,
                          default_dtype_np=np.float32,
                          load_graph=not process_only)

        self.dataset = dataset
        if not process_only:
            # check target
            self.dataset_info = DatasetInfo.from_dataset(dataset)

        return dataset

    def build_model(self):
        """
        DeepHE3Kernel build model process
        """
        # it is recommended to use save_model first and load model from there, instead of using build_model
        config = self.train_config

        num_species = len(self.dataset_info.index_to_z)
        print('Building model...')
        begin = time.time()
        net = Net(num_species=num_species,
                  irreps_embed_node=config.irreps_embed_node,
                  irreps_edge_init=config.irreps_edge_init,
                  irreps_sh=config.irreps_sh,
                  irreps_mid_node=config.irreps_mid_node,
                  irreps_post_node=config.irreps_post_node,
                  irreps_out_node=config.irreps_out_node,
                  irreps_mid_edge=config.irreps_mid_edge,
                  irreps_post_edge=config.irreps_post_edge,
                  irreps_out_edge=config.net_out_irreps,
                  num_block=config.num_blocks,
                  r_max=config.cutoff_radius,
                  use_sc=True,
                  no_parity=config.no_parity,
                  use_sbf=config.use_sbf,
                  only_ij=config.only_ij,
                  if_sort_irreps=False,
                  escn=True)

        print(f'Finished building model, cost {time.time() - begin:.2f} seconds.')
        self.net = net

        return net

    def register_constructor(self):
        """
        DeepHE3Kernel register_constructor
        """

        config = self.train_config

        construct_kernel = E3TensorDecompNet(config.net_out_irreps,
                                             self.net_out_info.js,
                                             default_dtype_ms=ms.float32,
                                             spinful=self.dataset_info.spinful,
                                             no_parity=config.no_parity,
                                             if_sort=config.convert_net_out)

        self.construct_kernel = construct_kernel

        return construct_kernel

    def config_set_target(self):
        """
        DeepHE3Kernel config_set_target
        """
        o, i, s = self.dataset_info.orbital_types, self.dataset_info.index_to_z, self.dataset_info.spinful
        self.train_config.set_target(o, i, s)
        self.net_out_info = NetOutInfo(self.train_config.target_blocks, self.dataset_info)

    def check_index_by_stru_id(self, stru_id, stru_id_place):
        """
        DeepHE3Kernel check_index_by_stru_id
        """
        for index, data in enumerate(self.dataset.data_numpy):
            if data[stru_id_place] == stru_id:
                return index
        return 0
    def get_loader(self):
        """
        DeepHE3Kernel get_loader
        """
        config = self.train_config
        dataset = self.dataset.data_numpy

        indices = list(range(len(dataset)))

        dataset_size = len(indices)
        train_size = int(config.train_ratio * dataset_size)
        val_size = int(config.val_ratio * dataset_size)
        if config.train_size > 0:
            train_size = config.train_size
        if config.val_size > 0:
            val_size = config.val_size

        np.random.shuffle(indices)
        dataset_tuple = tuple(dataset)

        print(f'size of train set: {len(indices[:train_size])}')
        generator_dataset_train = ds.GeneratorDataset(dataset_tuple,
                                                      column_names=[
                                                          "x", "edge_index", "edge_attr", "stru_id", "pos", "lattice",
                                                          "edge_key", "atom_num_orbital", "spinful", "label", "mask",
                                                          "mask_length", "mask_dim1", "mask_dim2", "mask_dim3"
                                                      ],
                                                      sampler=ds.SubsetRandomSampler(indices[:train_size]))

        val_indices = indices[train_size:train_size + val_size]
        print(f'size of val set: {len(val_indices)}')

        generator_dataset_val = ds.GeneratorDataset(dataset_tuple,
                                                    column_names=[
                                                        "x", "edge_index", "edge_attr", "stru_id", "pos", "lattice",
                                                        "edge_key", "atom_num_orbital", "spinful", "label", "mask",
                                                        "mask_length", "mask_dim1", "mask_dim2", "mask_dim3"
                                                    ],
                                                    sampler=ds.SubsetRandomSampler(val_indices))


        print(f'Batch size: {config.batch_size}')

        return generator_dataset_train, generator_dataset_val
