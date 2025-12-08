# Copyright 2025 Huawei Technologies Co., Ltd
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md
"""load data 'batch' used in test"""
import pickle
import mindspore as ms
from alphafold3.model.feat_batch import Batch


def load_batch(dtype=ms.float32):
    """Load batch data for test"""
    with open('/data/zmmVol2/AF3/test/unit_tests/model/diffusion/example_np.pkl', 'rb') as f:
        data = pickle.load(f)
    batch = Batch.from_data_dict(data)
    batch.convert_to_tensor(dtype=dtype)
    return batch
