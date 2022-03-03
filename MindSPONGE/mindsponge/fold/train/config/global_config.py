# Copyright 2022 Huawei Technologies Co., Ltd & CPL YiQin GAO Research Group
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

"""Global Model config."""

import copy

import ml_collections


def global_config(length: int) -> ml_collections.ConfigDict:
    """Get the global config."""
    if str(length) not in GLOBAL_CONFIG:
        raise ValueError(f'Invalid padding sequence length {length}.')
    cfg = copy.deepcopy(GLOBAL_CONFIG[str(length)])
    return cfg


GLOBAL_CONFIG = ml_collections.ConfigDict({
    "256": {
        'zero_init': True,
        'seq_length': 256,
        'extra_msa_length': 1024,
        'template_embedding': {
            'slice_num': 0,
        },
        'template_pair_stack': {
            'triangle_attention_starting_node': {
                'slice_num': 0,
            },
            'triangle_attention_ending_node': {
                'slice_num': 0,
            },
            'pair_transition': {
                'slice_num': 0,
            },
        },
        'extra_msa_stack': {
            'msa_transition': {
                'slice_num': 0,
            },
            'msa_row_attention_with_pair_bias': {
                'slice_num': 0,
            },
            'msa_column_global_attention': {
                'slice_num': 0,
            },
            'outer_product_mean': {
                'slice_num': 0,
            },
            'triangle_attention_starting_node': {
                'slice_num': 0,
            },
            'triangle_attention_ending_node': {
                'slice_num': 0,
            },
            'pair_transition': {
                'slice_num': 0,
            },
        },
        'evoformer_iteration': {
            'msa_transition': {
                'slice_num': 0,
            },
            'msa_row_attention_with_pair_bias': {
                'slice_num': 0,
            },
            'msa_column_attention': {
                'slice_num': 0,
            },
            'outer_product_mean': {
                'slice_num': 0,
            },
            'triangle_attention_starting_node': {
                'slice_num': 0,
            },
            'triangle_attention_ending_node': {
                'slice_num': 0,
            },
            'pair_transition': {
                'slice_num': 0,
            },
        },
    },
    "384": {
        'zero_init': True,
        'seq_length': 384,
        'extra_msa_length': 5120,
        'template_embedding': {
            'slice_num': 0,
        },
        'template_pair_stack': {
            'triangle_attention_starting_node': {
                'slice_num': 0,
            },
            'triangle_attention_ending_node': {
                'slice_num': 0,
            },
            'pair_transition': {
                'slice_num': 0,
            },
        },
        'extra_msa_stack': {
            'msa_transition': {
                'slice_num': 0,
            },
            'msa_row_attention_with_pair_bias': {
                'slice_num': 8,
            },
            'msa_column_global_attention': {
                'slice_num': 0,
            },
            'outer_product_mean': {
                'slice_num': 0,
            },
            'triangle_attention_starting_node': {
                'slice_num': 0,
            },
            'triangle_attention_ending_node': {
                'slice_num': 0,
            },
            'pair_transition': {
                'slice_num': 0,
            },
        },
        'evoformer_iteration': {
            'msa_transition': {
                'slice_num': 0,
            },
            'msa_row_attention_with_pair_bias': {
                'slice_num': 0,
            },
            'msa_column_attention': {
                'slice_num': 0,
            },
            'outer_product_mean': {
                'slice_num': 0,
            },
            'triangle_attention_starting_node': {
                'slice_num': 0,
            },
            'triangle_attention_ending_node': {
                'slice_num': 0,
            },
            'pair_transition': {
                'slice_num': 0,
            },
        },
    },
    "512": {
        'zero_init': True,
        'seq_length': 512,
        'extra_msa_length': 5120,
        'template_embedding': {
            'slice_num': 0,
        },
        'template_pair_stack': {
            'triangle_attention_starting_node': {
                'slice_num': 0,
            },
            'triangle_attention_ending_node': {
                'slice_num': 0,
            },
            'pair_transition': {
                'slice_num': 0,
            },
        },
        'extra_msa_stack': {
            'msa_transition': {
                'slice_num': 0,
            },
            'msa_row_attention_with_pair_bias': {
                'slice_num': 4,
            },
            'msa_column_global_attention': {
                'slice_num': 0,
            },
            'outer_product_mean': {
                'slice_num': 0,
            },
            'triangle_attention_starting_node': {
                'slice_num': 0,
            },
            'triangle_attention_ending_node': {
                'slice_num': 0,
            },
            'pair_transition': {
                'slice_num': 0,
            },
        },
        'evoformer_iteration': {
            'msa_transition': {
                'slice_num': 0,
            },
            'msa_row_attention_with_pair_bias': {
                'slice_num': 0,
            },
            'msa_column_attention': {
                'slice_num': 0,
            },
            'outer_product_mean': {
                'slice_num': 0,
            },
            'triangle_attention_starting_node': {
                'slice_num': 0,
            },
            'triangle_attention_ending_node': {
                'slice_num': 0,
            },
            'pair_transition': {
                'slice_num': 0,
            },
        },
    },
    "1024": {
        'zero_init': True,
        'seq_length': 1024,
        'extra_msa_length': 5120,
        'template_embedding': {
            'slice_num': 4,
        },
        'template_pair_stack': {
            'triangle_attention_starting_node': {
                'slice_num': 4,
            },
            'triangle_attention_ending_node': {
                'slice_num': 4,
            },
            'pair_transition': {
                'slice_num': 0,
            },
        },
        'extra_msa_stack': {
            'msa_transition': {
                'slice_num': 0,
            },
            'msa_row_attention_with_pair_bias': {
                'slice_num': 16,
            },
            'msa_column_global_attention': {
                'slice_num': 4,
            },
            'outer_product_mean': {
                'slice_num': 0,
            },
            'triangle_attention_starting_node': {
                'slice_num': 4,
            },
            'triangle_attention_ending_node': {
                'slice_num': 4,
            },
            'pair_transition': {
                'slice_num': 0,
            },
        },
        'evoformer_iteration': {
            'msa_transition': {
                'slice_num': 0,
            },
            'msa_row_attention_with_pair_bias': {
                'slice_num': 4,
            },
            'msa_column_attention': {
                'slice_num': 4,
            },
            'outer_product_mean': {
                'slice_num': 0,
            },
            'triangle_attention_starting_node': {
                'slice_num': 4,
            },
            'triangle_attention_ending_node': {
                'slice_num': 4,
            },
            'pair_transition': {
                'slice_num': 0,
            },
        },
    },
    "2048": {
        'zero_init': True,
        'seq_length': 2048,
        'extra_msa_length': 5120,
        'template_embedding': {
            'slice_num': 32,
        },
        'template_pair_stack': {
            'triangle_attention_starting_node': {
                'slice_num': 32,
            },
            'triangle_attention_ending_node': {
                'slice_num': 32,
            },
            'pair_transition': {
                'slice_num': 16,
            },
        },

        'extra_msa_stack': {
            'msa_transition': {
                'slice_num': 16,
            },
            'msa_row_attention_with_pair_bias': {
                'slice_num': 128,
            },
            'msa_column_global_attention': {
                'slice_num': 32,
            },
            'outer_product_mean': {
                'slice_num': 16,
            },
            'triangle_attention_starting_node': {
                'slice_num': 32,
            },
            'triangle_attention_ending_node': {
                'slice_num': 32,
            },
            'pair_transition': {
                'slice_num': 16,
            },
        },
        'evoformer_iteration': {
            'msa_transition': {
                'slice_num': 16,
            },
            'msa_row_attention_with_pair_bias': {
                'slice_num': 32,
            },
            'msa_column_attention': {
                'slice_num': 32,
            },
            'outer_product_mean': {
                'slice_num': 16,
            },
            'triangle_attention_starting_node': {
                'slice_num': 32,
            },
            'triangle_attention_ending_node': {
                'slice_num': 32,
            },
            'pair_transition': {
                'slice_num': 16,
            },
        },
    },
    "2304": {
        'zero_init': True,
        'seq_length': 2304,
        'extra_msa_length': 5120,
        'template_embedding': {
            'slice_num': 64,
        },
        'template_pair_stack': {
            'triangle_attention_starting_node': {
                'slice_num': 64,
            },
            'triangle_attention_ending_node': {
                'slice_num': 64,
            },
            'pair_transition': {
                'slice_num': 2,
            },
        },

        'extra_msa_stack': {
            'msa_transition': {
                'slice_num': 2,
            },
            'msa_row_attention_with_pair_bias': {
                'slice_num': 64,
            },
            'msa_column_global_attention': {
                'slice_num': 64,
            },
            'outer_product_mean': {
                'slice_num': 8,
            },
            'triangle_attention_starting_node': {
                'slice_num': 64,
            },
            'triangle_attention_ending_node': {
                'slice_num': 64,
            },
            'pair_transition': {
                'slice_num': 4,
            },
        },
        'evoformer_iteration': {
            'msa_transition': {
                'slice_num': 2,
            },
            'msa_row_attention_with_pair_bias': {
                'slice_num': 64,
            },
            'msa_column_attention': {
                'slice_num': 64,
            },
            'outer_product_mean': {
                'slice_num': 8,
            },
            'triangle_attention_starting_node': {
                'slice_num': 64,
            },
            'triangle_attention_ending_node': {
                'slice_num': 64,
            },
            'pair_transition': {
                'slice_num': 4,
            },
        },
    },
})
