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
"""Base trainer of downstream tasks."""
from abc import abstractmethod

from mindspore import nn, value_and_grad
from mindformers import T5Tokenizer
from mindformers.tools.logger import get_logger


logger = get_logger(logger_name='DownstreamTask')
PRINT_STEPS = 80


def lr_secheduler(init_lr, batch_num, epochs):
    """Cosine decay learning rate"""
    lr_max = init_lr  # max lr
    lr_min = 5e-5  # min lr
    decay_steps = int(epochs * batch_num)
    lr_sch = nn.CosineDecayLR(min_lr=lr_min, max_lr=lr_max, decay_steps=decay_steps)
    return lr_sch


class BaseTask:
    """
    Base class of downstream tasks with train and eval interface.

    Args:
        config.cate_name: the dataset has two subtask with different label name: 'loc', 'membrane'
        config.t5_config_path: prot t5 pretrain model directory path.
        config.checkpoint_path: the task checkpoint path; use to eval.
    """
    def __init__(self, config):
        self.mode = config.mode
        self.task_name = config.task_name
        self.t5_config_path = config.t5_config_path
        self.checkpoint_path = config.checkpoint_path

        self.train_conf = config.train
        self.checkpoint_save_path = config.train.checkpoint_save_path
        self.epochs = config.train.epochs

        self.net = None
        self.train_dataset = None
        self.eval_dataset = None
        self.grad_fn = None
        self.t5_tokenizer = T5Tokenizer.from_pretrained(config.t5_config_path)

    @abstractmethod
    def eval_acc(self, eval_data_path):
        pass

    @abstractmethod
    def forward_fn(self, *args):
        pass

    @abstractmethod
    def eval_fn(self, dataset):
        pass

    def train_step(self, *args):
        """train step"""
        loss, grads = self.grad_fn(*args)
        self.optimizer(grads)
        return loss

    def train(self):
        """train"""
        weights = self.net.trainable_params()
        self.grad_fn = value_and_grad(self.forward_fn, None, weights)

        logger.info("Begin training...")
        for epoch in range(self.epochs):
            logger.info("Epoch: %d", epoch)
            step = 0
            loss_steps = 0.0
            for inputs in self.train_dataset:
                step += 1
                loss = self.train_step(*inputs)
                loss_steps += loss.asnumpy()
                if step % PRINT_STEPS == 0:
                    logger.info("loss: %.4f", loss_steps / PRINT_STEPS)
                    loss_steps = 0.0

            self.train_dataset.reset()

        logger.info("Training done")

        if self.eval_dataset:
            logger.info("Begin eval...")
            acc = self.eval_fn(self.eval_dataset)
            logger.info("Accuracy: %s", str(acc))

        if self.checkpoint_save_path:
            self.net.save_checkpoint(self.checkpoint_save_path)
            logger.info("Checkpoint dumpped successful")
