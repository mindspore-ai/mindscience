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
"""T5 model with initial method."""
import logging

import mindspore
from mindspore.common.initializer import initializer, Constant, Normal

from mindformers import AutoConfig
from mindformers import T5ForConditionalGeneration as T5WithLoss
from mindformers.modules.transformer.transformer import default_transformer_config, TransformerOpParallelConfig

logger = logging.getLogger(__name__)


def set_data(weight, init_distribution):
    """set data weight"""
    weight.set_data(initializer(init_distribution, weight.shape, weight.dtype))


def init_cell(cell, name, config):
    """init cell"""
    factor = config.initializer_factor
    if "layernorm" in name:
        set_data(cell.gamma, Constant(factor * 1.0))
    elif "tfm_embedding_lookup" in name:
        # Mesh TensorFlow embeddings initialization
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L1624
        set_data(cell.embedding_table, Normal(factor * 1.0))
    elif name.endswith("output"):
        # default these cells has no bias
        set_data(cell.mapping.weight, Normal(factor * ((config.hidden_size) ** -0.5)))
        set_data(cell.projection.weight, Normal(factor * ((config.d_ff) ** -0.5)))
    elif name.endswith("attention"):
        d_model = config.hidden_size
        key_value_proj_dim = config.kv_size
        n_heads = config.num_heads

        # q, k, v, o parameter
        set_data(cell.dense1.weight, Normal(factor * ((d_model * key_value_proj_dim) ** -0.5)))
        set_data(cell.dense2.weight, Normal(factor * (d_model**-0.5)))
        set_data(cell.dense3.weight, Normal(factor * (d_model**-0.5)))
        set_data(cell.projection.weight, Normal(factor * ((n_heads * key_value_proj_dim) ** -0.5)))

        if cell.has_relative_bias and cell.is_cross_atten:
            set_data(cell.cross_bias, Normal(factor * (d_model**-0.5)))
    else:
        # weights that do not require change
        pass


def init_t5_weights(cell, config, prefix=''):
    """init t5 weights"""
    if hasattr(cell, 'add_name'):
        return

    cell.add_flags(add_name=prefix)
    init_cell(cell, prefix, config)

    for name, sub_cell in cell.cells_and_names():
        hier_name = prefix + "." + name
        init_t5_weights(sub_cell, config, prefix=hier_name)


def trans_to_transformer_config(parallel_config):
    """trans_to_transformer_config"""
    if not parallel_config:
        return default_transformer_config

    return TransformerOpParallelConfig(**parallel_config)


def create_model(config_path, load_model_path=None, parallel_config=None, from_pretrained=False):
    """create model"""
    if from_pretrained:
        return T5WithLoss.from_pretrained(config_path)

    base_config = AutoConfig.from_pretrained(config_path)
    base_config.parallel_config = trans_to_transformer_config(parallel_config)
    model = T5WithLoss(base_config)

    if load_model_path:
        # load from checkpoint path
        param_dict = mindspore.load_checkpoint(load_model_path)
        mindspore.load_param_into_net(model, param_dict)
        logger.info("pretrain: load ckpt successful")
    else:
        # init T5
        init_t5_weights(model, base_config, prefix="")
        logger.info("pretrain: inited successful")

    return model
