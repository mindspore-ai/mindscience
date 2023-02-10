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
    files = os.listdir(path)
    for file in files:
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            dir_walk(file_path, file_list)
        else:
            file_list.append(file_path)


class PSP(DataSet):
    """PSP DataSet"""
    def __init__(self):

        self.url = {
            "train": ["http://ftp.cbi.pku.edu.cn/psp/true_structure_dataset/",
                      "http://ftp.cbi.pku.edu.cn/psp/distillation_dataset/"],
            "validation": ["http://ftp.cbi.pku.edu.cn/psp/new_validation_dataset/"],
            "examples": ["https://download.mindspore.cn/mindscience/mindsponge/MEGAFold/examples/"]}

        self.cache = "./psp_data/"
        self.in_memory = False
        super().__init__()
        self.mode = ["train", "validation", "examples"]

    def __getitem__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def download(self, path=None, mode="validation"):
        """download"""
        if path is not None:
            self.cache = path
        # WARNING: just for linux OS
        print("Start download data for mode : ", mode)
        for url in self.url[mode]:
            command = "wget -c -r -np -k -L -p -P " + self.cache + " " + url
            os.system(command)

        file_list = []
        dir_walk(self.cache, file_list)
        tar_gz_list = []
        for val in file_list:
            if val.endswith("tar.gz"):
                tar_gz_list.append(val)

        print("Start uncompression ... ")
        for i in tqdm(range(len(tar_gz_list))):
            val = tar_gz_list[i]
            short_path, _ = os.path.split(val.split("/psp/")[-1])
            dir_path = os.path.join(self.cache, short_path)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            tar_file = tarfile.open(val)
            tar_file.extractall(dir_path)
        print("Finish uncompression ... ")
        print("PSP DataSet has been saved in ", self.cache)
        if mode == "train":
            print("Make training name list")
            self.make_name_list()

    def make_name_list(self):
        pass

    def process(self):
        raise NotImplementedError

    def data_parse(self, input, idx):
        raise NotImplementedError

    def create_iterator(self):
        raise NotImplementedError
