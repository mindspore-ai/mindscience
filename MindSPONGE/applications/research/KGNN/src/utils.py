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
"""utils"""
import os
import stat
import json
import pickle

from src.model_utils.config import config

if config.enable_modelarts:
    import moxing as mox


def pickle_load(filename: str):
    """load pickle file"""
    try:
        flags = os.O_RDONLY
        modes = stat.S_IWUSR | stat.S_IRUSR
        with os.fdopen(os.open(filename, flags, modes), 'rb') as fout:
            obj = pickle.load(fout)
        print(f'Logging Info - Loaded: {filename}')
    except EOFError:
        print(f'Logging Error - Cannot load: {filename}')
        obj = None

    return obj


def pickle_dump(filename: str, obj):
    """save python object to pickle file"""
    flags = os.O_WRONLY | os.O_CREAT
    modes = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open(filename, flags, modes), 'wb') as fout:
        pickle.dump(obj, fout)
    print(f'Logging Info - Saved: {filename}')


def write_log(filename: str, log, mode='w'):
    """write log"""
    flags = os.O_WRONLY | os.O_CREAT
    modes = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open(filename, flags, modes), mode) as writers:
        writers.write('\n')
        json.dump(log, writers, indent=4, ensure_ascii=False)


def format_filename(base_dir: str, filename_template: str, **kwargs):
    """Obtain the filename of data based on the provided template and parameters"""
    filename = os.path.join(base_dir, filename_template.format(**kwargs))
    return filename


def obs_env(obs_data_url, data_dir):
    """Copy single dataset from obs to training image"""
    mox.file.copy_parallel(obs_data_url, data_dir)
    print("Successfully Download {} to {}".format(obs_data_url, data_dir))


def obs_url_env(obs_ckpt_url, ckpt_url):
    """
    Copy ckpt file from obs to inference image
    To operate on folders, use mox.file.copy_parallel. If copying a file.
    Please use mox.file.copy to operate the file, this operation is to operate the file
    """
    mox.file.copy(obs_ckpt_url, ckpt_url)
    print("Successfully Download {} to {}".format(obs_ckpt_url, ckpt_url))


def env_obs(train_dir, obs_train_url):
    """Copy the output to obs"""
    mox.file.copy_parallel(train_dir, obs_train_url)
    print("Successfully Upload {} to {}".format(train_dir, obs_train_url))


def download_qizhi(obs_data_url, data_dir):
    """DownloadFromQizhi"""
    obs_env(obs_data_url, data_dir)


def upload_qizhi(train_dir, obs_train_url):
    env_obs(train_dir, obs_train_url)
