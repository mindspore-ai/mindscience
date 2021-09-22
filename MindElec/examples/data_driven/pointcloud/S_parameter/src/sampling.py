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

DATA_CONFIG_PATH = "./data_config.npz"
def generate_random_data():
    np.save("input.npy", np.ones((100, 496, 20, 40, 3), np.float32))
    np.save("data.npy", np.ones((100, 1001), np.float32))
    np.save(DATA_CONFIG_PATH, scale_s11=1)

if __name__ == "__main__":
    generate_random_data()
