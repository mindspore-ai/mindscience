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
# ==============================================================================
"""
prepare dataset
"""
import numpy as np
import pandas as pd

from mindelec.data import Dataset, ExistedDataConfig
from mindelec.geometry import Cuboid
from mindelec.geometry import create_config_from_edict

from src.config import cuboid_sampling_config


def preprocessing_waveguide_data(waveguide_data_path, npy_save_path,
                                 label_save_path):
    """
    Convert the data with format .csv from Modulus to .npy.
    """
    df = pd.read_csv(waveguide_data_path)
    print(df.head())
    # The .csv file contains 8 columns, the first two columns is (x, y) coordinates,
    # the last six columns (u0, u1, ..., u5) are the measure value under different
    # conditions. Here just use u0 as the example, which follows the Modulus example.
    df["z"] = -0.5 * np.ones(len(df))
    df_need = df[["z", "x", "y"]]
    print(df_need.head())
    points = df_need.to_numpy(dtype=np.float32)
    label = df[[f"u{i}" for i in range(6)]].to_numpy(dtype=np.float32)

    print(points[:5, :])
    print(label[:5, :])
    np.save(npy_save_path, points)
    np.save(label_save_path, label)


def create_train_dataset(config) -> Dataset:
    """
    create trainning dataset from existed data and sample
    """
    # The left (waveguide plane) data
    npy_points_path = config["waveguide_points_path"]
    waveguide_port = ExistedDataConfig(name=config["waveguide_name"],
                                       data_dir=[npy_points_path],
                                       columns_list=["points"],
                                       data_format="npy",
                                       constraint_type="Label",
                                       random_merge=False)
    # Other faces and domain data
    cuboid_space = Cuboid(name=config["geom_name"],
                          coord_min=config["coord_min"],
                          coord_max=config["coord_max"],
                          sampling_config=create_config_from_edict(cuboid_sampling_config))
    geom_dict = {cuboid_space: ["domain", "BC"]}

    # create dataset for train and test
    train_dataset = Dataset(geom_dict, existed_data_list=[waveguide_port])

    return train_dataset
