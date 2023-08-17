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
# ============================================================================
"""grover"""
import mindspore as ms
from mindspore import jit, nn, context
from mindspore.common import mutable
from ..model import Model
from .nn_arch import GROVEREmbedding, GroverFinetuneTask, GroverFpGenerationTask, GroverPretrainTask
from .src.util.scheduler import get_lr


def load_parameters(network, file_name):
    """
    Load parameters for evaluating.
    """
    param_dict = ms.load_checkpoint(file_name)
    param_dict_new = {}
    filter_key = {}
    for key, values in param_dict.items():
        if key.startswith('grover.') or key.startswith('mol'):
            if key in filter_key:
                continue
            param_dict_new[key] = values
        else:
            continue
    ms.load_param_into_net(network, param_dict_new)


def load_convert_params(args, network):
    """
    Load pretrained model parameters for finetuning.
    """
    if args.resume_grover:
        param_dict = ms.load_checkpoint(args.resume_grover)
        param_dict_new = {}
        for key, values in param_dict.items():
            param_dict_new[key] = values
        ms.load_param_into_net(network, param_dict_new)


def eval_context_init(args):
    """
    Init context.
    """
    ms.set_context(device_target=args.device_target, save_graphs=False)


def gen_context_init(args):
    """
    Init Context.
    """
    ms.set_context(device_target=args.device_target, save_graphs=False)

    if ms.get_context("device_target") == "Ascend":
        ms.set_context(max_device_memory="10GB")


def pretrain_context_init(args):
    """
    Init context.
    """
    ms.set_context(device_target=args.device_target, save_graphs=False)

    if ms.get_context("device_target") == "Ascend":
        ms.set_context(max_device_memory="10GB")


def train_context_init(args):
    """
    Init context.
    """
    ms.set_context(device_target=args.device_target, save_graphs=False)

    if ms.get_context("device_target") == "Ascend":
        ms.set_context(max_device_memory="10GB")


class Grover(Model):
    """Grover"""
    name = "Grover"

    def __init__(self, config, **kwargs):
        self.config = config
        self.use_jit = self.config.use_jit
        if context.get_context("device_target") == "GPU":
            self.mixed_precision = False
        else:
            self.mixed_precision = True
        self.checkpoint_url = 'https://download.mindspore.cn/mindscience/mindsponge/grover/checkpoint/grover.ckpt'
        self.checkpoint_path = "./grover.ckpt"
        if self.config.parser_name == "eval":
            eval_context_init(config)
            config.is_training = False
            grover_model = GROVEREmbedding(config)
            network = GroverFinetuneTask(config, grover_model, is_training=config.is_training)
            network.set_train(False)
        elif self.config.parser_name == "gen":
            gen_context_init(config)
            config.is_training = False
            grover_model = GROVEREmbedding(config)
            network = GroverFpGenerationTask(config, grover_model)
            network.set_train(False)
        elif self.config.parser_name == "pretrain":
            pretrain_context_init(config)
            config.is_training = True
            grover_model = GROVEREmbedding(config)
            network = GroverPretrainTask(config, grover_model,
                                         atom_vocab_size=kwargs['atom_vocab_size'],
                                         bond_vocab_size=kwargs['bond_vocab_size'],
                                         fg_size=kwargs['fg_size'])
            config.steps_per_epoch = kwargs['steps_per_epoch']
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
        else:
            train_context_init(config)
            config.is_training = True
            config.features_dim = kwargs['features_dim']
            config.output_size = kwargs['output_size']
            grover_model = GROVEREmbedding(config)
            network = GroverFinetuneTask(config, grover_model, is_training=config.is_training)
            config.steps_per_epoch = kwargs['steps_per_epoch']
            lr = get_lr(config)
            opt = nn.Adam(network.trainable_params(), learning_rate=ms.Tensor(lr), weight_decay=config.weight_decay)
            if config.mixed:
                loss_scale_manager = ms.FixedLossScaleManager(config.loss_scale_value, drop_overflow_update=False)
                network = ms.build_train_network(network, optimizer=opt, loss_scale_manager=loss_scale_manager,
                                                 level="O2", keep_batchnorm_fp32=False)

                for _, cell in network.cells_and_names():
                    if isinstance(cell, (GroverFinetuneLossBlock, nn.Softmax, nn.LayerNorm, SelectIndex)):
                        cell.to_float(ms.float32)
            else:
                network = nn.TrainOneStepCell(network=network, optimizer=opt)
            network.set_train(True)
        self.network = network
        super().__init__(self.checkpoint_url, self.checkpoint_path, self.network, self.name,
                         mixed_precision=self.mixed_precision)

    # pylint: disable=arguments-differ
    def forward(self, input_graph, scope, features_batch):
        if self.use_jit:
            # pylint: disable=arguments-differ
            preds = self._jit_forward(input_graph, scope, features_batch)
        else:
            preds = self._pynative_forward(input_graph, scope, features_batch)
        return preds

    def predict(self, data, **kwargs):
        preds = None
        if self.config.parser_name == "eval":
            features_batch = data["features"]
            f_atoms = data["f_atoms"]
            f_bonds = data["f_bonds"]
            a2b = data["a2b"]
            b2a = data["b2a"]
            b2revb = data["b2revb"]
            a2a = data["a2a"]
            a_scope = data["a_scope"].asnumpy().tolist()
            b_scope = data["b_scope"].asnumpy().tolist()
            scope = (a_scope, b_scope)
            input_graph = (f_atoms, f_bonds, a2b, b2a, b2revb, a2a)
            preds = self.forward(input_graph, scope, features_batch)
        else:
            features_batch = data["features"]
            a_scope = data["a_scope"].asnumpy().tolist()
            b_scope = data["b_scope"].asnumpy().tolist()
            scope = (a_scope, b_scope)
            input_graph = (data["f_atoms"], data["f_bonds"], data["a2b"], data["b2a"], data["b2revb"], data["a2a"])
            preds = self.forward(input_graph, scope, features_batch)
        return preds

    def loss(self, data):
        pass

    def grad_operations(self, gradient):
        pass

    @jit
    def backward(self, data):
        loss = self.network(*data)
        return loss

    def train_step(self, data):
        if self.config.parser_name == "pretrain":
            a_scope = data["a_scope"].asnumpy().tolist()
            b_scope = data["b_scope"].asnumpy().tolist()
            scope = (a_scope, b_scope)
            input_graph = (data["f_atoms"], data["f_bonds"], data["a2b"], data["b2a"], data["b2revb"], data["a2a"])
            input_graph = mutable(input_graph)
            targets = (data["atom_vocab_label"], data["bond_vocab_label"], data["fgroup_label"])
            targets = mutable(targets)
            feat = (input_graph, scope, targets)
        else:
            features_batch = data["features"]
            targets = data["labels"]
            a_scope = data["a_scope"].asnumpy().tolist()
            b_scope = data["b_scope"].asnumpy().tolist()
            scope = (a_scope, b_scope)
            input_graph = (data["f_atoms"], data["f_bonds"], data["a2b"], data["b2a"], data["b2revb"], data["a2a"])
            input_graph = mutable(input_graph)
            feat = (input_graph, scope, features_batch, targets)
        loss = self.backward(feat)
        return loss

    # pylint: disable=arguments-differ
    def _pynative_forward(self, input_graph, scope, features_batch):
        preds = self.network(input_graph, scope, features_batch)
        return preds

    # pylint: disable=arguments-differ
    @jit
    def _jit_forward(self, data, scope, features_batch):
        preds = self.network(data, scope, features_batch)
        return preds
