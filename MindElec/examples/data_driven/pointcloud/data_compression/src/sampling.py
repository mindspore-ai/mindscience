# Copyright 2021 Huawei Technologies Co., Ltd
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
"""pseudo data generation"""


import numpy as np

DATA_CONFIG_PATH = "./data_config.npy"
def generate_random_data():
    np.save("train_data.npy", np.ones((1000, 5, 25, 50, 25), np.float32))
    np.save("test_data.npy", np.ones((20, 5, 25, 50, 25), np.float32))
    np.save(DATA_CONFIG_PATH, np.ones((4), np.float32))

if __name__ == "__main__":
    generate_random_data()
