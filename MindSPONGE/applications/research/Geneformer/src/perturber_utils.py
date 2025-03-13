# Copyright 2025 Huawei Technologies Co., Ltd
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

"""perturber utiles script"""
import os
import re
import glob
from multiprocessing import Pool
from datasets import load_from_disk

from mindformers import AutoConfig, BertForTokenClassification


def flatten_list(megalist):
    """flatten list"""
    return [item for sublist in megalist for item in sublist]


def parse_filename(filename):
    """parse filename"""
    match = re.search(r"mindformers_rank_(\d+)-(\d+)_(\d+)\.ckpt", filename)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    return None


def filter_data_by_criteria(example, criteria):
    """filter data by criteria"""
    return example[criteria['key']] in criteria['value']


def filter_by_dict(data, filter_data, nproc):
    """filter by dict"""
    criteria_list = [{'key': key, 'value': value} for key, value in filter_data.items()]
    with Pool(nproc) as pool:
        results = pool.starmap(
            filter_data_by_criteria,
            [(example, criteria) for example in data for criteria in criteria_list]
            )
    filtered_results = [all(result) for result in zip(*results)]
    data = data[filtered_results]
    return data


def load_and_filter(filter_data, nproc, input_data_file):
    """load and filter"""
    data = load_from_disk(input_data_file)
    if filter_data:
        data = filter_by_dict(data, filter_data, nproc)
    return data


def quant_layers(model):
    """quant layers"""
    layer_nums = []
    for name, _ in model.parameters_and_names():
        if name.endswith(".attention.projection.weight"):
            layer_nums.append(int(name.split(".attention.projection.weight")[0].split(".")[-1]))
    return max(layer_nums) + 1


def find_latest_ckpt(directory):
    """find latest ckpt"""
    ckpt_files = glob.glob(os.path.join(directory, '*.ckpt'))
    if not ckpt_files:
        return None
    latest_ckpt = None
    latest_time = 0
    for ckpt_file in ckpt_files:
        creation_time = os.path.getctime(ckpt_file)
        if creation_time > latest_time:
            latest_time = creation_time
            latest_ckpt = ckpt_file
    return latest_ckpt


def load_model(model_directory, mode, config_path="config/run_geneformer_args.yaml"):
    """load model weights"""
    geneformer_config = AutoConfig.from_pretrained(config_path)
    if not os.path.exists(os.path.join(model_directory, "geneformer_mindspore.ckpt")):
        raise FileNotFoundError(os.path.join(model_directory, "geneformer_mindspore.ckpt") + " not found")
    geneformer_config.load_checkpoint = os.path.join(model_directory, "geneformer_mindspore.ckpt")
    geneformer_config.checkpoint_name_or_path = os.path.join(model_directory, "geneformer_mindspore.ckpt")
    model = BertForTokenClassification(geneformer_config)
    if mode == "eval":
        model.eval()
    return model
