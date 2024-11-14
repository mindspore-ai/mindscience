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
"""ProtT5 Trainer with data parallel."""
import time
import os

import mindspore as ms
from mindspore import nn, value_and_grad
from mindspore.amp import all_finite
from mindspore.ops import functional as F
from mindspore.parallel._utils import _get_device_num, _get_gradients_mean
from mindspore.communication import init, get_rank, get_group_size
from mindformers.core.clip_grad import ClipGradNorm
from mindformers.tools.logger import get_logger
from mindformers import T5Tokenizer

from .optimization import create_optimizer, WarmUpPolynomialDecayLR
from .t5_dataloader import create_pretrain_dataset, find_mindrecord_files
from .t5_modeling import create_model
from ..utils.utils import generate_checkpoint_filename
from ...model import Model

PRINT_ITERS = 10
logger = get_logger(logger_name='Pretrain')


class ProtT5(Model):
    """ProtT5"""
    name = "ProtT5"

    def __init__(self, config):
        self.config = config
        self.use_parallel = config.parallel
        self.rank_id = 0
        self.rank_size = 1
        self.init_context()

        self.checkpoint_url = "https://download.mindspore.cn/mindscience/mindsponge/ProtT5/checkpoint/prot_t5_xl.ckpt"
        self.checkpoint_path = "./prot_t5_xl.ckpt"
        self.mode = config.mode
        self.train_conf = config.train

        if self.mode == "train":
            self.network = create_model(config.t5_config_path, config.load_model_path)
            self.init_trainer()
        else:
            self.tokenizer = T5Tokenizer.from_pretrained(config.t5_config_path)
            self.network = create_model(config.t5_config_path, from_pretrained=True)

        super().__init__(self.checkpoint_url, self.checkpoint_path, self.network, self.name, None,
                         mixed_precision=False)


    def init_context(self):
        """init context"""
        if self.use_parallel:
            init()
            self.rank_id = get_rank()
            self.rank_size = get_group_size()
            ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=self.rank_id)
            ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL, parameter_broadcast=True,
                                         device_num=self.rank_size, gradients_mean=True)

        else:
            ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend")


    def init_trainer(self):
        """init trainer"""
        if self.train_conf.save_ckpt_path:
            os.makedirs(self.train_conf.save_ckpt_path, exist_ok=True)

        # data_loader
        dataset_path = find_mindrecord_files(self.train_conf.train_data_path)
        train_dataset = create_pretrain_dataset(dataset_path, self.train_conf.batch_size, self.train_conf.epochs, \
                                                rank_size=self.rank_size, rank_id=self.rank_id)

        self.train_dataloader = train_dataset.create_tuple_iterator()
        num_train_steps = train_dataset.get_dataset_size()

        # grad clip
        self.use_clip_grad = False
        if self.train_conf.use_clip_grad:
            self.use_clip_grad = True
            self.clip_grad_norm = ClipGradNorm(max_norm=self.train_conf.max_grad_norm)

         # trick: warm up
        if self.train_conf.warmup_steps > 0:
            lr = WarmUpPolynomialDecayLR(self.train_conf.lr, 0.0, self.train_conf.warmup_steps, num_train_steps, 1.0)
        else:
            lr = self.train_conf.lr

        # Define optimizer.
        self.optimizer = create_optimizer(self.network, lr, 'adam', weight_decay=0)

        # data parall
        if self.use_parallel:
            degree = _get_device_num()
            mean = _get_gradients_mean()
            self.grad_reducer = nn.DistributedGradReducer(self.optimizer.parameters, mean, degree)

        weights = self.network.trainable_params()
        self.grad_fn = value_and_grad(self.forward_fn, None, weights, has_aux=False)

    def forward_fn(self, input_ids, input_mask, decode_ids):
        """forward loss"""
        loss = self.network(input_ids, input_mask, decode_ids)
        return loss

    def save_checkpoint(self, train_step_nums):
        """save checkpoint"""
        if self.rank_id == 0:
            filename = generate_checkpoint_filename(self.train_conf.save_ckpt_path, train_step_nums)
            ms.save_checkpoint(self.network, filename)

    def train(self):
        """train"""
        loss_total = 0
        cur_step_nums, train_step_nums, skip_step_nums = 0, 0, 0
        cur_time, avg_time = time.time(), 0

        # step begin
        self.network.set_train(True)

        for input_ids, input_mask, decode_ids in self.train_dataloader:
            loss, is_finite = self._train_step(input_ids, input_mask, decode_ids)
            if is_finite:
                loss_total = loss_total + loss.asnumpy().item()
                train_step_nums += 1
            else:
                logger.warning(f"grads overflow, skip step {cur_step_nums}; loss: {loss}")
                skip_step_nums += 1

            if train_step_nums % PRINT_ITERS == 0 and train_step_nums != 0:
                print_time = time.time()
                total_time = print_time - cur_time
                cur_time = print_time
                avg_time = total_time / (PRINT_ITERS + skip_step_nums)

                logger.info(f"avg_time(ms): {avg_time * 1000:2f}, "
                            f"cur_step: {cur_step_nums}, "
                            f"skip_steps: {skip_step_nums:3d}, "
                            f"train_step: {train_step_nums}, "
                            f"loss: {loss_total/PRINT_ITERS:f}, ")

                loss_total = 0
                skip_step_nums = 0

            # saving ckpt per N steps or last step
            if train_step_nums % self.train_conf.save_steps == 0:
                self.save_checkpoint(train_step_nums)

            cur_step_nums += 1

        self.save_checkpoint(train_step_nums)
        logger.info("Pretrain done!")


    @ms.jit
    def _train_step(self, input_ids, input_mask, decode_ids):
        """train step jit function"""
        loss, grads = self.grad_fn(input_ids, input_mask, decode_ids)

        if self.use_parallel:
            grads = self.grad_reducer(grads)

        is_finite = all_finite(grads)

        if is_finite:
            # Apply gradient clipping
            if self.use_clip_grad:
                grads, _ = self.clip_grad_norm(grads)

            loss = F.depend(loss, self.optimizer(grads))

        return loss, is_finite

    def train_step(self, data):
        """train step"""
        return self._train_step(*data)

    # pylint: disable=W0221
    def predict(self, data, mode="embedding"):
        """predict"""
        self.network.set_train(False)
        token_ids, attention_mask = data
        if mode == "generate":
            # Generate the sequence of input texts
            output_ids = self.network.generate(token_ids, do_sample=False)
            output_tokens = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            return output_tokens

        if mode == "embedding":
            # Embedding of the final layer of encoder
            outputs = self.network.encoder_forward(token_ids, attention_mask)
            hiddens = outputs.asnumpy()
            return hiddens

        return None

    def forward(self, data):
        pass

    def backward(self, data):
        pass

    def _jit_forward(self, data):
        pass

    def _pynative_forward(self, data):
        pass
