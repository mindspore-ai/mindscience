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
"""Create dataset."""
from copy import deepcopy

from mindflow.data import Dataset
from mindflow.geometry import Rectangle, generate_sampling_config


def create_dataset(config, n_samps=None):
    """Create dataset."""
    if n_samps is not None:
        config = deepcopy(config)
        config["data"]["domain"]["size"] = n_samps
        config["data"]["BC"]["size"] = n_samps
        config["batch_size"] = n_samps
    sampling_config = generate_sampling_config(config["data"])
    region = Rectangle(
        "rectangle", **config["geometry"]["rectangle"], sampling_config=sampling_config
    )
    dataset = Dataset({region: ["domain", "BC"]})
    ds_create = dataset.create_dataset(
        batch_size=config["batch_size"],
        shuffle=True,
        prebatched_data=True,
        drop_remainder=True,
    )
    return ds_create
