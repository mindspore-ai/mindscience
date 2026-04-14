# Copyright 2022 Huawei Technologies Co., Ltd
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
"""dataset"""
from math import pow
import numpy as np
import mindspore as ms
import mindspore.ops as ops

def pad_and_cat(data, batchinfo):
    # data: list[dict{rho: Tensor, v: Tensor}]
    # batchinfo: dict{batch_size: int, input_channel: int}
    data = {k: ms.Tensor(np.stack([d[k] for d in data], axis=0)) for k in data[0].keys()}
    return data

class DensityToPotentialDataset:
    def __init__(self, raw_data, input_channel=4):
        super(DensityToPotentialDataset, self).__init__()
        self.data_np = raw_data.astype(np.float32)
        self.input_channel = input_channel

    def __len__(self):
        return len(self.data_np)

    def __getitem__(self, idx):
        if isinstance(idx, slice): 
            raise ValueError("slice may cause problem in DensityToPotentialDataset.__getitem__()")
        data_slice = np.asarray(self.data_np[slice(idx, idx + 1, None)])
        ndim = int(round(pow(float(data_slice.shape[1] - 1) / float(self.input_channel),  1. / 3)))
        rho = data_slice[:, :-1].reshape(self.input_channel, ndim, ndim, ndim)
        v = data_slice[:, -1]
        return {"rho": rho, "v": v}


class DensityToPotentialDataset_zsym:
    def __init__(self, raw_data, input_channel=4):
        super(DensityToPotentialDataset_zsym, self).__init__()
        self.data_np = raw_data
        self.input_channel = input_channel

    def __len__(self):
        return len(self.data_np)

    def __getitem__(self, idx):
        if isinstance(idx, slice): 
            raise ValueError("slice may cause problem in DensityToPotentialDataset.__getitem__()")
        data_slice = np.asarray(self.data_np[slice(idx, idx + 1, None)])
        ndim = int(round(pow(float(data_slice.shape[1] - 1) / float(self.input_channel),  1. / 3)))

        rho_zp = data_slice[:, :-1].reshape(self.input_channel, ndim, ndim, ndim)
        rho_zm = np.empty_like(rho_zp)
        for ic in range(self.input_channel):
            for iz in range(ndim):
                rho_zm[ic, :, :, iz] = rho_zp[ic, :, :, -(iz+1)]
        rho_zm[-1, :, :, :] *= -1
        
        rho = np.array([rho_zp, rho_zm])
        assert(rho.shape == (2, self.input_channel, ndim, ndim, ndim))

        v = data_slice[:, -1]
        return {"rho": rho, "v": v}


def get_train_and_validate_set(options, toDataLoader=True):
    raw_data = np.load(options["data_path"])
    np.random.shuffle(raw_data)

    constrain = options["constrain"]
    if constrain.find("zsym") != -1:
        train_set = DensityToPotentialDataset_zsym(
                raw_data[:options["train_set_size"]], 
                input_channel=options["input_channel"])
        validate_set = DensityToPotentialDataset_zsym(
            raw_data[options["train_set_size"] : options["train_set_size"] + options["validate_set_size"]], 
            input_channel=options["input_channel"])

    else:
        # constrain == "none"
        train_set = DensityToPotentialDataset(
                raw_data[:options["train_set_size"]], 
                input_channel=options["input_channel"])
        validate_set = DensityToPotentialDataset(
            raw_data[options["train_set_size"] : options["train_set_size"] + options["validate_set_size"]], 
            input_channel=options["input_channel"])

    if not toDataLoader: return train_set, validate_set

    train_set = ms.dataset.GeneratorDataset(train_set, column_names=["data"], shuffle=options["shuffle"])
    train_set_loader = train_set.batch(
            batch_size=int(options["batch_size"]), 
            per_batch_map=pad_and_cat,
            num_parallel_workers=int(options["num_workers"]))

    validate_set = ms.dataset.GeneratorDataset(validate_set, column_names=["data"], shuffle=options["shuffle"])
    validate_set_loader = validate_set.batch(
            batch_size=int(options["batch_size"]),
            per_batch_map=pad_and_cat,
            num_parallel_workers=int(options["num_workers"]))

    return train_set_loader, validate_set_loader
    
def get_test_set(options, toDataLoader=True):
    raw_data = np.load(options["data_path"])
    
    constrain = options["constrain"]
    if constrain.find("zsym") != -1:
        test_set = DensityToPotentialDataset_zsym(
                raw_data[:options["test_set_size"]], 
                input_channel=options["input_channel"])
    else:
        test_set = DensityToPotentialDataset(
                raw_data[:options["test_set_size"]], 
                input_channel=options["input_channel"])

    if not toDataLoader: return test_set

    test_set = ms.dataset.GeneratorDataset(test_set, ["data"], shuffle=options["shuffle"])
    test_set_loader = test_set.batch(
            batch_size=1,
            per_batch_map=pad_and_cat,
            num_parallel_workers=int(options["num_workers"]))

    return test_set_loader
    
