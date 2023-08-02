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
"""
construct dataset based on saved mindrecord
"""
import os
import numpy as np
import mindspore.dataset as ds
from mindspore.mindrecord import FileWriter

from .pde_solvers import initgen, VariantCoeLinear2d


class DataGenerator():
    """
    Generate train or test data and save data in mindrecord files.
    """

    def __init__(self, step, mode, data_size, file_name, config):
        self.size = data_size
        self.scale = config["solver_mesh_scale"]
        self.output_mesh_size = config["mesh_size"]
        self.mesh_size = self.output_mesh_size * self.scale
        if isinstance(self.mesh_size, int):
            self.mesh_size = [self.mesh_size] * 2
        self.enable_noise = config["enable_noise"]
        self.start_noise_level = config["start_noise_level"]
        self.end_noise_level = config["end_noise_level"]

        self.initfreq = config["init_freq"]
        self.boundary = 'Periodic'
        self.dt = config["dt"]

        self.step = step
        self.mode = mode
        self.file_name = file_name

        self.spectral_size = 30
        self.max_dt = 5e-3
        self.variant_coe_magnitude = config["variant_coe_magnitude"]
        self.pde = VariantCoeLinear2d(spectral_size=self.spectral_size, max_dt=self.max_dt,
                                      variant_coe_magnitude=self.variant_coe_magnitude)

    def process(self):
        """generate data and save as mindrecord"""
        data_writer = FileWriter(file_name=self.file_name, shard_num=1, overwrite=True)
        schema_json = {}
        schema_json["u0"] = {"type": "float32", "shape": [self.output_mesh_size, self.output_mesh_size]}

        if self.mode == "train":
            schema_json["uT"] = {"type": "float32", "shape": [self.output_mesh_size, self.output_mesh_size]}
        elif self.mode == "test":
            for i in range(self.step):
                schema_json["u_step{}".format(i + 1)] = {"type": "float32",
                                                         "shape": [self.output_mesh_size, self.output_mesh_size]}
        data_writer.add_schema(schema_json, "data_schema")

        for i in range(self.size):
            samples = []
            sample = {}
            if self.mode == "train":
                u = self._generate_data_train_mode()
                u = self._post_process(u)
                sample["u0"] = u[0, ...]
                sample["uT"] = u[1, ...]
            else:
                samples = []
                sample = {}
                u = self._generate_data_test_mode()
                u = self._post_process(u)
                sample["u0"] = u[0, ...]
                for j in range(self.step):
                    sample["u_step{}".format(j + 1)] = u[j + 1, ...]
            samples.append(sample)
            data_writer.write_raw_data(samples)
        data_writer.commit()

        print("Mindrecorder saved")

    def _generate_data_test_mode(self):
        """generate data in test process"""
        time_steps = [i * self.dt for i in range(1, self.step + 1)]
        init = initgen(mesh_size=self.mesh_size, freq=self.initfreq, boundary=self.boundary)
        u = np.zeros([self.step + 1] + list(init.shape))
        u[0, :, :] = init[:, :]
        cur_u = init
        time_steps = [0] + time_steps
        for i in range(len(time_steps) - 1):
            cur_u = self.pde.predict(cur_u, time_steps[i + 1] - time_steps[i])
            u[i + 1, :, :] = cur_u[:, :]
        return u

    def _generate_data_train_mode(self):
        """generate data in train process"""
        init = initgen(mesh_size=self.mesh_size, freq=self.initfreq, boundary=self.boundary)
        u = np.zeros([2] + list(init.shape))
        u[0, :, :] = init[:, :]
        time_step = self.step * self.dt
        u[1, :, :] = self.pde.predict(init, time_step)
        return u

    def _post_process(self, u):
        """sampling and add noise"""
        if self.boundary == 'Periodic':
            idx1 = slice(np.random.randint(self.scale), None, self.scale)
            idx2 = slice(np.random.randint(self.scale), None, self.scale)
        else:
            idx1 = slice(self.scale - 1, None, self.scale)
            idx2 = slice(self.scale - 1, None, self.scale)

        u = u[:, idx1, idx2]
        u = u.reshape(-1, 1, self.output_mesh_size, self.output_mesh_size)

        noise = np.zeros(list(u.shape))
        if self.enable_noise:
            stdvar = np.sqrt(np.mean((u[0, 0, :, :] - np.mean(u[0, 0, :, :])) ** 2))
            noise[0, 0, :, :] = self.start_noise_level * stdvar * np.random.randn(self.output_mesh_size,
                                                                                  self.output_mesh_size)
            for i in range(1, u.shape[0]):
                noise[i, 0, :, :] = self.end_noise_level * stdvar * np.random.randn(self.output_mesh_size,
                                                                                    self.output_mesh_size)
        u += noise
        return u


class DataPrepare():
    """Obtain dataset for train or test from mindrecord."""

    def __init__(self, config, data_file):
        self.mesh_size = config["mesh_size"]
        self.batch_size = config["batch_size"]
        self.data_file = data_file

    def create_test_dataset(self, step):
        dataset = ds.MindDataset(dataset_files=self.data_file, shuffle=True,
                                 columns_list=["u0", "u_step{}".format(step)])
        dataset = dataset.batch(batch_size=1)
        return dataset

    def create_train_dataset(self):
        dataset = ds.MindDataset(dataset_files=self.data_file, shuffle=True, columns_list=["u0", "uT"])
        dataset = dataset.batch(batch_size=self.batch_size, drop_remainder=True)
        operations = [lambda x, y: (
            x.reshape(-1, 1, 1, self.mesh_size, self.mesh_size), y.reshape(-1, 1, self.mesh_size, self.mesh_size))]
        dataset = dataset.map(operations, input_columns=["u0", "uT"])
        dataset_train, dataset_eval = dataset.split([0.5, 0.5])
        return dataset_train, dataset_eval


def create_dataset(config, step, db_name, mode, data_size=0):
    """crate dataset"""
    file_name = os.path.join(config["mindrecord_data_dir"], db_name)
    data = DataGenerator(step=step, config=config, mode=mode, data_size=data_size, file_name=file_name)
    data.process()
    dataset = DataPrepare(config=config, data_file=file_name)
    return dataset
