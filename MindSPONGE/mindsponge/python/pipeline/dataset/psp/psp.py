# Copyright 2023 The AIMM Group at Shenzhen Bay Laboratory & Peking University & Huawei Technologies Co., Ltd
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
"""PSP"""
import os
import tarfile
from tqdm import tqdm
from ..dataset import DataSet


def dir_walk(path, file_list):
    "dir_walk"
    files = os.listdir(path)
    for file in files:
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            dir_walk(file_path, file_list)
        else:
            file_list.append(file_path)

TRAIN_URL = ["http://ftp.cbi.pku.edu.cn/psp/true_structure_dataset/pdb/" + f"pdb_{i}.tar.gz" \
             for i in range(256)] + \
            ["http://ftp.cbi.pku.edu.cn/psp/true_structure_dataset/pkl/" + f"pkl_{i}.tar.gz" \
             for i in range(256)] + \
            ["http://ftp.cbi.pku.edu.cn/psp/distillation_dataset/pdb/" + f"pdb_{i}.tar.gz" \
             for i in range(256)] + \
            ["http://ftp.cbi.pku.edu.cn/psp/distillation_dataset/pkl/" + f"pkl_{i}.tar.gz" \
             for i in range(256)] + \
            ["http://ftp.cbi.pku.edu.cn/psp/true_structure_dataset/true_structure_data_statistics_729.json",
             "http://ftp.cbi.pku.edu.cn/psp/distillation_dataset/distill_data_statistics_729.json"]
EXAMPLE_URL = ["http://ftp.cbi.pku.edu.cn/psp/true_structure_dataset/true_structure_data_statistics_729.json",
               "http://ftp.cbi.pku.edu.cn/psp/true_structure_dataset/pdb/pdb_0.tar.gz",
               "http://ftp.cbi.pku.edu.cn/psp/true_structure_dataset/pkl/pkl_0.tar.gz"]
VALIDATION_URL = ["http://ftp.cbi.pku.edu.cn/psp/new_validation_dataset/pdb.tar.gz",
                  "http://ftp.cbi.pku.edu.cn/psp/new_validation_dataset/pkl.tar.gz",
                  "http://ftp.cbi.pku.edu.cn/psp/new_validation_dataset/nv_data_statistics.json"]


class PSP(DataSet):
    """PSP DataSet"""
    def __init__(self):

        self.url = {
            "train": TRAIN_URL,
            "train_examples": EXAMPLE_URL,
            "validation": VALIDATION_URL,
            "examples": ["https://download.mindspore.cn/mindscience/mindsponge/MEGAFold/examples/"]}

        self.cache = "./psp_data/"
        self.pkl_path = "./psp_data/pkl/"
        self.pdb_path = "./psp_data/pdb/"
        self.in_memory = False
        super().__init__()

    def __getitem__(self, **kwargs):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    @property
    def dataset_path(self):
        return self.cache

    def download(self, path=None, mode="examples"):
        if path is not None:
            self.cache = path
        if not os.path.exists(self.cache):
            os.mkdir(self.cache)
        if mode not in self.url:
            raise KeyError(f"Only {self.url.keys()} \
                           are supported as PSP dataset mode, but got {mode}")
        print("Start download data for mode : ", mode)
        for url in self.url[mode]:
            command = "wget -P " + self.cache + " " + url
            os.system(command)

        file_name_list = os.listdir(self.cache)
        print("Start uncompression ... ")
        for i in tqdm(range(len(file_name_list))):
            val = file_name_list[i]
            if  not val.endswith("tar.gz"):
                continue
            for file_type in ["pkl", "pdb"]:
                if file_type in val:
                    dir_path = os.path.join(self.cache, file_type)
                    if not os.path.exists(dir_path):
                        os.makedirs(dir_path)
                    tar_file = tarfile.open(os.path.join(self.cache, val))
                    tar_file.extractall(dir_path)
                    sub_path = dir_path + "/" + val.split(".")[0]
                    os.system(f"rename _renum.pdb .pdb {sub_path}/*")
                    os.system(f"mv {sub_path}/* {dir_path}")
                    os.system(f"rm -rf {sub_path}")

        print("Finish uncompression ... ")
        print("PSP DataSet has been saved in ", self.cache)


    def make_name_list(self):
        "make_name_list"
        pkl_names = os.listdir(os.path.join(self.cache, "pkl"))
        pkl_names = [name.split(".")[0] for name in pkl_names if name[-4:] == ".pkl"]
        pdb_names = os.listdir(os.path.join(self.cache, "pdb"))
        pdb_names = [name.split(".")[0] for name in pdb_names if name[-4:] == ".pdb"]
        name_list = list(set(pkl_names).intersection(set(pdb_names)))
        return name_list

    def process(self, data, **kwargs):
        raise NotImplementedError


    def data_parse(self, idx):
        raise NotImplementedError


    def create_iterator(self, num_epochs, **kwargs):
        raise NotImplementedError

    def _generate_probability(self, data_statistics):
        for key, value in data_statistics.items():
            length_prob = max(256, min(512, value["sequence_length"])) / 512.0
            cluster_prob = 1.0 / float(value.get("cluster_size", 1))
            data_statistics[key]["probs"] = length_prob * cluster_prob
