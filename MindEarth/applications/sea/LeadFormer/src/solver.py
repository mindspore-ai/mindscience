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
"""train"""
import os

import numpy as np

import mindspore
import mindspore.nn as nn
from mindspore import Model, context
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from .segformer import SegFormer
from .data_loader import create_multi_class_dataset
from .loss import EvolutionLoss
from .utils import StepLossTimeMonitor, filter_checkpoint_parameter_by_list
from .myop import AdamWeightDecay


def learning_rate_function(lr_all, cur_step_num):
    return lr_all[cur_step_num]


class Trainer(nn.Cell):
    """Neural network trainer class that encapsulates the complete training pipeline

    This class is responsible for:
    - Initializing training configurations and model
    - Handling distributed training setup
    - Managing dataset loading and preprocessing
    - Configuring optimizer and learning rate schedule
    - Executing training loop and saving checkpoints

    Attributes:
        epochs (int): Total number of training epochs
        rank (int): Process rank in distributed training
        group_size (int): Total number of processes in distributed training
        data_dir (str): Path to training data directory
        run_distribute (bool): Whether to enable distributed training
        model_name (str): Name of the model being used
        net (nn.Cell): Neural network model instance
        resume (bool): Whether to resume training from checkpoint
        resume_ckpt (str): Path to checkpoint for resuming training
        transfer_training (bool): Whether to perform transfer learning
        filter_weight (list): List of parameter names to filter during transfer learning
        repeat (int): Number of dataset repetitions
        split (float): Train/validation split ratio
        num_classes (int): Number of classes in classification task
        train_augment (bool): Whether to enable training data augmentation
        output_path (str): Path to save training outputs
        keep_checkpoint_max (int): Maximum number of checkpoints to keep
        weight_decay (float): Weight decay coefficient for optimizer
        batch_size (int): Training batch size
        lr (float): Initial learning rate
        amp_level (str): Auto mixed precision level (O0/O1/O2/O3)
    """
    def __init__(self, config, epochs=400):
        super().__init__()
        self.epochs = epochs
        self.rank = 0
        self.group_size = 1
        self.data_dir = config["data"].get("data_path", "")
        self.run_distribute = config["train"].get("run_distribute", False)
        self.model_name = config["model"].get("name", "")
        self.in_channels = config["model"].get("in_channels", [64, 128, 320, 512])
        if self.model_name == "ice_simple":
            self.net = SegFormer(
                in_channels=self.in_channels,
                num_classes=1,
                embedding_dim=256,
            )
        else:
            raise ValueError("Unsupported model: {}".format(self.model_name))
        self.resume = config["train"].get("resume", False)
        self.resume_ckpt = config["train"].get("resume_ckpt", "./")
        self.transfer_training = config["train"].get("transfer_training", False)
        self.filter_weight = config["model"].get("filter_weight", [])
        self.repeat = config["train"].get("repeat", 1)
        self.split = config["data"].get("split", 0.98)
        self.num_classes = config["model"].get("num_classes", 1)
        self.train_augment = config["data"].get("train_augment", False)
        self.output_path = config["summary"].get("output_path", "./train")
        self.keep_checkpoint_max = config["summary"].get("keep_checkpoint_max", 1)
        self.weight_decay = config["optimizer"].get("weight_decay", 0.01)
        self.batch_size = config["data"].get("batch_size", 1)
        self.lr = config["optimizer"].get("lr", 0.0001)
        self.amp_level = config["train"].get("amp_level", "O3")

    def train(self):
        """train"""
        if self.run_distribute:
            init()
            self.group_size = get_group_size()
            self.rank = get_rank()
            parallel_mode = ParallelMode.DATA_PARALLEL
            context.set_auto_parallel_context(
                parallel_mode=parallel_mode,
                device_num=self.group_size,
                gradients_mean=False,
            )

        if self.resume:
            param_dict = load_checkpoint(self.resume_ckpt)
            if self.transfer_training:
                filter_checkpoint_parameter_by_list(param_dict, self.filter_weight)
            load_param_into_net(self.net, param_dict)

        dataset_sink_mode = False
        per_print_times = 1
        train_dataset = create_multi_class_dataset(
            self.data_dir,
            self.repeat,
            self.batch_size,
            is_train=True,
            split=self.split,
            rank=self.rank,
            group_size=self.group_size,
            shuffle=True,
        )
        train_data_size = train_dataset.get_dataset_size()
        ckpt_save_dir = os.path.join(self.output_path, f"ckpt_{self.rank}")
        save_ck_steps = train_data_size
        ckpt_config = CheckpointConfig(
            save_checkpoint_steps=save_ck_steps,
            keep_checkpoint_max=self.keep_checkpoint_max,
        )
        ckpoint_cb = ModelCheckpoint(
            prefix="ckpt_{}_adam".format(self.model_name),
            directory=ckpt_save_dir,
            config=ckpt_config,
        )

        end_learning_rate = 0.00
        step_per_epoch = train_data_size
        total_step = int(step_per_epoch * self.epochs / self.repeat) + 1
        decay_epoch = int(self.epochs / self.repeat)
        exponential_decay_lr = np.array(
            nn.cosine_decay_lr(
                end_learning_rate, self.lr, total_step, step_per_epoch, decay_epoch
            )
        )
        optimizer = AdamWeightDecay(
            params=self.net.trainable_params(),
            learning_rate=self.lr,
            beta1=0.9,
            beta2=0.999,
            weight_decay=self.weight_decay,
        )
        loss_scale = mindspore.train.loss_scale_manager.FixedLossScaleManager(
            loss_scale=2048
        )
        criterion = EvolutionLoss(self.net)
        model = Model(
            self.net,
            loss_fn=criterion,
            loss_scale_manager=loss_scale,
            optimizer=optimizer,
            amp_level=self.amp_level,
        )
        print("============== Starting Training ==============")
        callbacks = [
            StepLossTimeMonitor(
                lr_all=exponential_decay_lr,
                learning_rate_func=learning_rate_function,
                batch_size=self.batch_size,
                per_print_times=per_print_times,
                rank=self.rank,
            ),
            ckpoint_cb,
        ]
        print("==============================================================")
        model.train(
            int(self.epochs / self.repeat),
            train_dataset,
            callbacks=callbacks,
            dataset_sink_mode=dataset_sink_mode,
        )
        print("============== End Training ==============")
