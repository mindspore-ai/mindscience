# Copyright 2025 Huawei Technologies Co., Ltd
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
init
"""
import os

from mindspore import ops, Tensor, nn, load_param_into_net, load_checkpoint, save_checkpoint


class SciModule(nn.Cell):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def load(self, ckpt_file):
        assert ckpt_file.endswith('.ckpt') and os.path.exists(ckpt_file)
        param_dict = load_checkpoint(ckpt_file)
        load_param_into_net(self, param_dict)
        print(f"Load checkpoint from {ckpt_file}")


    def save(self, ckpt_file):
        """保存模型参数到 checkpoint 文件"""
        os.makedirs(os.path.dirname(ckpt_file), exist_ok=True)
        save_checkpoint(self, ckpt_file)

    def set_grad(self, special_layer_patterns=None):
        """对符合特殊名称模式的网络层设置requires_grad"""
        super().set_grad()
        # 默认匹配包含"encoder"或"decoder"的层
        special_layer_patterns = special_layer_patterns or ["encoder", "decoder"]
        for param in self.parameters():
            param.requires_grad = any(pattern in param.name for pattern in special_layer_patterns)

    @property
    def num_params(self):
        return sum([i.numel() for i in self.trainable_params()])

    def set_precision(self, precision=mindspore.float32, special_layer_patterns=None, blacklist_patterns=None, whitelist_patterns=None):
        # 设置默认黑白名单
        blacklist = blacklist_patterns or ["softmax", "layernorm", "batchnorm"]
        whitelist = whitelist_patterns or []
        
        for name, cell in self.cells_and_names():
            # 黑名单检查：强制float32
            if any(pattern in name.lower() for pattern in blacklist):
                cell.to_float(mindspore.float32)
                for param in cell.get_parameters():
                    param.set_data(param.data.astype(mindspore.float32))
            # 白名单检查：应用用户指定精度
            elif any(pattern in name.lower() for pattern in whitelist):
                cell.to_float(precision)
                for param in cell.get_parameters():
                    param.set_data(param.data.astype(precision))
            # 其他层：按special_layer_patterns处理
            elif special_layer_patterns and any(pattern in name.lower() for pattern in special_layer_patterns):
                cell.to_float(precision)
                for param in cell.get_parameters():
                    param.set_data(param.data.astype(precision))