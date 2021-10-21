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
"""dataset utilities"""

from mindelec.data import Dataset, ExistedDataConfig

def create_dataset(input_path, label_path, batch_size=8, shuffle=True):
    electromagnetic = ExistedDataConfig(name="electromagnetic",
                                        data_dir=[input_path, label_path],
                                        columns_list=["inputs", "label"],
                                        data_format=input_path.split('.')[-1])
    dataset = Dataset(existed_data_list=[electromagnetic])
    data_loader = dataset.create_dataset(batch_size=batch_size, shuffle=shuffle)

    return data_loader
