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

"""Moxing adapter for ModelArts"""

import os
import stat
import time
import json
from src.model_utils.config import config

if config.enable_modelarts:
    import moxing as mox


def get_device_id():
    device_id = os.getenv('DEVICE_ID', '0')
    return int(device_id)


def get_device_num():
    device_num = os.getenv('RANK_SIZE', '1')
    return int(device_num)


def get_rank_id():
    global_rank_id = os.getenv('RANK_ID', '0')
    return int(global_rank_id)


def get_job_id():
    job_id = os.getenv('JOB_ID')
    job_id = job_id if job_id != "" else "default"
    return job_id


# Copy single dataset from obs to training image
def obs_to_env(obs_data_url, data_dir):
    """Copy single dataset from obs to training image"""
    try:
        mox.file.copy_parallel(obs_data_url, data_dir)
        print("Successfully Download {} to {}".format(obs_data_url, data_dir))
    except IOError as e:
        print('moxing download {} to {} failed: '.format(obs_data_url, data_dir) + str(e))
    # Set a cache file to determine whether the data has been copied to obs.
    # If this file exists during multi-card training, there is no need to copy the dataset multiple times.
    flags = os.O_WRONLY | os.O_CREAT
    modes = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open('/cache/download_input.txt', flags, modes), 'w') as f:
        f.close()
    try:
        if os.path.exists("/cache/download_input.txt"):
            print("download_input succeed")
    except IOError as e:
        print("download_input failed")


def multi_obs_to_env(multi_data_url, data_dir):
    """Copy multi-single dataset from obs to training image"""
    # --multi_data_url is json data, need to do json parsing for multi_data_url
    multi_data_json = json.loads(multi_data_url)
    for i, _ in enumerate(multi_data_json):
        path = data_dir + "/" + multi_data_json[i]["dataset_name"]
        if not os.path.exists(path):
            os.makedirs(path)
        try:
            mox.file.copy_parallel(multi_data_json[i]["dataset_url"], path)
            print("Successfully Download {} to {}".format(multi_data_json[i]["dataset_url"], path))
        except IOError as e:
            print('moxing download {} to {} failed: '.format(
                multi_data_json[i]["dataset_url"], path) + str(e))
    # Set a cache file to determine whether the data has been copied to obs.
    # If this file exists during multi-card training, there is no need to copy the dataset multiple times.

    flags = os.O_WRONLY | os.O_CREAT
    modes = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open('/cache/download_input.txt', flags, modes), 'w') as f:
        f.close()
    try:
        if os.path.exists("/cache/download_input.txt"):
            print("download_input succeed")
    except IOError as e:
        print("download_input failed")


def obs_url_to_env(obs_ckpt_url, ckpt_url):
    """
    Copy ckpt file from obs to inference image
    To operate on folders, use mox.file.copy_parallel. If copying a file.
    Please use mox.file.copy to operate the file, this operation is to operate the file
    """
    try:
        mox.file.copy(obs_ckpt_url, ckpt_url)
        print("Successfully Download {} to {}".format(obs_ckpt_url, ckpt_url))
    except IOError as e:
        print('moxing download {} to {} failed: '.format(obs_ckpt_url, ckpt_url) + str(e))


# Copy the output to obs
def env_to_obs(train_dir, obs_train_url):
    """Copy the output to obs"""
    try:
        mox.file.copy_parallel(train_dir, obs_train_url)
        print("Successfully Upload {} to {}".format(train_dir, obs_train_url))
    except IOError as e:
        print('moxing upload {} to {} failed: '.format(train_dir, obs_train_url) + str(e))


def download_from_qizhi_multi(multi_data_url, data_dir):
    """Download multi-files from qizhi"""
    device_num = int(os.getenv('RANK_SIZE'))
    if device_num == 1:
        multi_obs_to_env(multi_data_url, data_dir)
    if device_num > 1:
        # Copying obs data does not need to be executed multiple times, just let the 0th card copy the data
        local_rank = int(os.getenv('RANK_ID'))
        if local_rank % 8 == 0:
            multi_obs_to_env(multi_data_url, data_dir)
        # If the cache file does not exist, it means that the copy data has not been completed,
        # and Wait for 0th card to finish copying data
        while not os.path.exists("/cache/download_input.txt"):
            time.sleep(1)


def download_from_qizhi(obs_data_url, data_dir):
    """DownloadFromQizhi"""
    device_num = int(os.getenv('RANK_SIZE'))
    if device_num == 1:
        obs_to_env(obs_data_url, data_dir)
    if device_num > 1:
        # Copying obs data does not need to be executed multiple times, just let the 0th card copy the data
        local_rank = int(os.getenv('RANK_ID'))
        if local_rank % 8 == 0:
            obs_to_env(obs_data_url, data_dir)
        # If the cache file does not exist, it means that the copy data has not been completed,
        # and Wait for 0th card to finish copying data
        while not os.path.exists("/cache/download_input.txt"):
            time.sleep(1)


def upload_to_qizhi(train_dir, obs_train_url):
    """
    Upload files to qizhi.
    """
    device_num = int(os.getenv('RANK_SIZE'))
    local_rank = int(os.getenv('RANK_ID'))
    if device_num == 1:
        env_to_obs(train_dir, obs_train_url)
    if device_num > 1:
        if local_rank % 8 == 0:
            env_to_obs(train_dir, obs_train_url)
