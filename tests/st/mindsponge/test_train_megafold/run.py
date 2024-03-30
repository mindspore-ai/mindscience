
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
"""megafold train script"""
import argparse
import mindspore as ms
from mindsponge import PipeLine


EPOCH_NUM = {
    "910A": 800,
    "910B": 1000
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='run megafold')
    parser.add_argument('--device_type', default="910A", type=str, help='device type')
    arguments = parser.parse_args()
    train_data_path = "./MEGA-Protein/"
    model_name = 'MEGAFold'
    ms.set_context(mode=ms.GRAPH_MODE)
    pipe = PipeLine(name=model_name)
    device_type = arguments.device_type
    pipe.initialize(config_path=f'./config/training_{device_type}.yaml')
    num_epochs = EPOCH_NUM[device_type]
    _ = pipe.train(train_data_path, num_epochs=num_epochs)
