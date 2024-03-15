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
parse_configs
"""

# pylint: disable=W0123

import os
import time
from configparser import ConfigParser
import numpy as np
import mindspore as ms
from mindchemistry.e3.o3.irreps import Irreps
from .utils import orbital_analysis


class BaseConfig:
    """
    BaseConfig class
    """

    def __init__(self, config_file=None):

        self._config = ConfigParser(inline_comment_prefixes=(';',))

        if self.__class__ is __class__:
            base_default = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'default_configs/base_default.ini')
            self.get_config(config_file, config_file_default=base_default)

            self.get_basic_section()
            self.get_data_section()

    def get_basic_section(self):
        """
        BaseConfig class get_basic_section process
        """
        dtype = self._config.get('basic', 'dtype')
        self.set_dtype(dtype)

    def get_data_section(self):
        """
        BaseConfig class get_data_section process
        """
        index_data = 'data'
        self.graph_dir = self._config.get(index_data, 'graph_dir')

        self.dft_data_dir = self._config.get(index_data, 'DFT_data_dir')
        self.processed_data_dir = self._config.get(index_data, 'processed_data_dir')

        self.save_graph_dir = self._config.get(index_data, 'save_graph_dir')
        self.target_data = self._config.get(index_data, 'target_data')
        self.dataset_name = self._config.get(index_data, 'dataset_name')

        self.get_olp = self._config.getboolean(index_data, 'get_overlap')

        self.only_ij = False
        self.ms_dtype = None
        self.np_dtype = None

    def get_config(self, config_file, config_file_default=''):
        """
        BaseConfig class get_config process
        """
        if config_file_default:
            self._config.read(config_file_default)
        self.config_file = config_file
        self._config.read(config_file)

    def set_dtype(self, dtype):
        """
        BaseConfig class set_dtype process
        """
        if dtype == 'float':
            self.ms_dtype = ms.float32
            self.np_dtype = np.float32
        elif dtype == 'float16':
            self.ms_dtype = ms.float16
            self.np_dtype = np.float16
        else:
            raise NotImplementedError


class TrainConfig(BaseConfig):
    """
    TrainConfig class
    """

    def __init__(self, config_file):
        super().__init__()
        train_default = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'default_configs/train_default.ini')

        self.get_config(config_file, config_file_default=train_default)

        self.get_basic_section()
        self.get_data_section()
        self.get_train_section()

        if self.use_new_hypp:
            self.get_hypp_section()

        if self.checkpoint_dir:
            config1_path = os.path.join(os.path.dirname(self.checkpoint_dir), 'src/train.ini')
            self.get_config(config1_path, config_file_default=train_default)

        # overwrite settings using those in config
        if not self.use_new_hypp:
            self.get_hypp_section()

        self.get_target_section()
        self.get_network_section()

        self._target_set_flag = False
        # set_target should be called once dataset has been prepared

    @property
    def target_blocks(self):
        """
        TrainConfig class get target_blocks
        """
        return self._target_blocks

    @property
    def net_out_irreps(self):
        """
        TrainConfig class get net_out_irreps
        """
        return self._net_out_irreps

    @property
    def irreps_post_edge(self):
        """
        TrainConfig class get irreps_post_edge
        """
        return self._irreps_post_edge

    @property
    def irreps_post_node(self):
        """
        TrainConfig class get irreps_post_node
        """
        ipn = self._config.get('network', 'irreps_post_node')
        if ipn:
            return ipn
        return self._irreps_post_node

    def get_basic_section(self):
        super().get_basic_section()
        index_basic = 'basic'
        self.seed = self._config.getint(index_basic, 'seed')
        self.checkpoint_dir = self._config.get(index_basic, 'checkpoint_dir')
        self.simp_out = self._config.getboolean(index_basic, 'simplified_output')
        self.use_new_hypp = self._config.getboolean(index_basic, 'use_new_hypp')

        # = save to time folder =
        save_dir = self._config.get(index_basic, 'save_dir')
        additional_folder_name = self._config.get(index_basic, 'additional_folder_name')
        self.save_dir = os.path.join(save_dir, str(time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))))
        if additional_folder_name:
            self.save_dir = self.save_dir + '_' + additional_folder_name

    def get_train_section(self):
        """
        TrainConfig class get_train_section process
        """
        index_train = 'train'
        self.batch_size = self._config.getint(index_train, 'batch_size')
        self.num_epoch = self._config.getint(index_train, 'num_epoch')

        self.min_lr = self._config.getfloat(index_train, 'min_lr')

        ev = self._config.get(index_train, 'extra_validation')
        self.extra_val = eval(ev) if ev else []
        self.extra_val_test_only = self._config.getboolean(index_train, 'extra_val_test_only')

        self.train_ratio = self._config.getfloat(index_train, 'train_ratio')
        self.val_ratio = self._config.getfloat(index_train, 'val_ratio')
        self.test_ratio = self._config.getfloat(index_train, 'test_ratio')

        self.train_size = self._config.getint(index_train, 'train_size')
        self.val_size = self._config.getint(index_train, 'val_size')
        self.test_size = self._config.getint(index_train, 'test_size')

    def get_hypp_section(self):
        """
        TrainConfig class get_hypp_section process
        """
        index_hyperparameters = 'hyperparameters'
        self.lr = self._config.getfloat(index_hyperparameters, 'learning_rate')
        self.adam_betas = eval(self._config.get(index_hyperparameters, 'Adam_betas'))

        self.scheduler_type = self._config.getint(index_hyperparameters, 'scheduler_type')
        ts = self._config.get(index_hyperparameters, 'scheduler_params')
        if ts:
            ts = 'dict' + ts
            self.scheduler_params = eval(ts)
        else:
            self.scheduler_params = dict()

        self.revert_decay_patience = self._config.getint(index_hyperparameters, 'revert_decay_patience')
        self.revert_decay_rate = self._config.getfloat(index_hyperparameters, 'revert_decay_rate')

    def get_target_section(self):
        """
        TrainConfig class get_target_section process
        """
        # target
        index_target = 'target'
        self.target = self._config.get(index_target, 'target')
        tbt = self._config.get(index_target, 'target_blocks_type')
        self.tbt0 = tbt[0].lower()
        self._target_blocks = None
        if self.tbt0 == 's':
            self._target_blocks = eval(self._config.get(index_target, 'target_blocks'))
        sep = self._config.get(index_target, 'selected_element_pairs')
        self.element_pairs = eval(sep) if sep else None
        self.convert_net_out = self._config.getboolean(index_target, 'convert_net_out')

    def get_network_section(self):
        """
        TrainConfig class get_network_section process
        """
        index_network = 'network'
        self.cutoff_radius = self._config.getfloat(index_network, 'cutoff_radius')
        self.only_ij = self._config.getboolean(index_network, 'only_ij')
        self.no_parity = self._config.getboolean(index_network, 'ignore_parity')

        sh_lmax = self._config.get(index_network, 'spherical_harmonics_lmax')
        sbf_irreps = self._config.get(index_network, 'spherical_basis_irreps')
        if sh_lmax:
            sh_lmax = int(sh_lmax)
            self.irreps_sh = Irreps([(1, (i, 1 if self.no_parity else (-1)**i)) for i in range(sh_lmax + 1)])
            self.use_sbf = False
        else:
            self.irreps_sh = Irreps(sbf_irreps)
            self.use_sbf = True

        irreps_mid = self._config.get(index_network, 'irreps_mid')
        if irreps_mid:
            self.irreps_mid_node = irreps_mid
            self._irreps_post_node = irreps_mid
            self.irreps_mid_edge = irreps_mid
        irreps_embed = self._config.get(index_network, 'irreps_embed')
        if irreps_embed:
            self.irreps_embed_node = irreps_embed
            self.irreps_edge_init = irreps_embed
        if self.target in ['hamiltonian']:
            self.irreps_out_node = '1x0e'
        self.num_blocks = self._config.getint(index_network, 'num_blocks')

        # ! post edge
        for name in ['irreps_embed_node', 'irreps_edge_init', 'irreps_mid_node', 'irreps_out_node', 'irreps_mid_edge']:
            irreps = self._config.get(index_network, name)
            if irreps:
                delattr(self, name)
                setattr(self, name, irreps)

        self._net_out_irreps = self._config.get(index_network, 'out_irreps')
        self._irreps_post_edge = self._config.get(index_network, 'irreps_post_edge')

    def set_target(self, orbital_types, index_to_z, spinful):
        """
        TrainConfig class set_target process
        """
        atom_orbitals = {}
        for z, orbital_type in zip(index_to_z, orbital_types):
            atom_orbitals[str(z.item())] = orbital_type

        target_blocks, net_out_irreps, irreps_post_edge = orbital_analysis(atom_orbitals,
                                                                           self.tbt0,
                                                                           spinful,
                                                                           targets=self._target_blocks,
                                                                           element_pairs=self.element_pairs,
                                                                           no_parity=self.no_parity)

        self._target_blocks = target_blocks
        if not self._net_out_irreps:
            self._net_out_irreps = net_out_irreps
        if not self._irreps_post_edge:
            self._irreps_post_edge = irreps_post_edge

        if self.convert_net_out:
            self._net_out_irreps = Irreps(self._net_out_irreps).sort().irreps.simplify()

        self._target_set_flag = True



class EvalConfig(BaseConfig):
    """
    EvalConfig class
    """

    def __init__(self, config_file):
        super().__init__()
        eval_default = (os.path.join(os.path.dirname(os.path.abspath(__file__)), 'default_configs/eval_default.ini'))

        self.get_config(config_file, config_file_default=eval_default)

        self.get_basic_section()
        self.get_data_section()

    def get_basic_section(self):
        """
        EvalConfig class get_basic_section process
        """
        index_basic = 'basic'
        self.model_dir = self._config.get(index_basic, 'trained_model_dir')
        super().get_basic_section()
        self.out_dir = self._config.get(index_basic, 'output_dir')
        os.makedirs(self.out_dir, exist_ok=True)
        self.target = self._config.get(index_basic, 'target')
        self.inference = self._config.getboolean(index_basic, 'inference')
        self.test_only = self._config.getboolean(index_basic, 'test_only')
