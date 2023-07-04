# Copyright 2021-2023 @ Shenzhen Bay Laboratory &
#                       Peking University &
#                       Huawei Technologies Co., Ltd
#
# This code is a part of AIchemist package.
#
# The AIchemist is open-source software based on the AI-framework:
# MindSpore (https://www.mindspore.cn/)
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
wn18
"""

import os
from .. import data
from .. import util


class WN18(data.KnowledgeGraphSet):
    """
    WordNet knowledge base.

    Statistics:
        - #Entity: 40,943
        - #Relation: 18
        - #Triplet: 151,442

    Parameters:
        path (str): path to store the dataset
        verbose (int, optional): output verbose level
    """

    urls = [
        "https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/raw/master/data/wn18/train.txt",
        "https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/raw/master/data/wn18/valid.txt",
        "https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/raw/master/data/wn18/test.txt",
    ]
    md5s = [
        "7d68324d293837ac165c3441a6c8b0eb",
        "f4f66fec0ca83b5ebe7ad7003404e61d",
        "b035247a8916c7ec3443fa949e1ff02c"
    ]

    def __init__(self, path, batch_size, verbose=1, shuffle=True, **kwargs):
        super().__init__(batch_size=batch_size, verbose=verbose, shuffle=shuffle, **kwargs)
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

    def load(self):
        """load"""
        txt_files = []
        for url, md5 in zip(self.urls, self.md5s):
            save_file = "wn18_" + os.path.basename(url)
            txt_file = util.download(url, self.path, save_file=save_file, md5=md5)
            txt_files.append(txt_file)

        self.load_file(txt_files)
        return self


class WN18RR(data.KnowledgeGraphSet):
    """
    A filtered version of WN18 dataset without trivial cases.

    Statistics:
        - #Entity: 40,943
        - #Relation: 11
        - #Triplet: 93,003

    Parameters:
        path (str): path to store the dataset
        verbose (int, optional): output verbose level
    """

    urls = [
        "https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/raw/master/data/wn18rr/train.txt",
        "https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/raw/master/data/wn18rr/valid.txt",
        "https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/raw/master/data/wn18rr/test.txt",
    ]
    md5s = [
        "35e81af3ae233327c52a87f23b30ad3c",
        "74a2ee9eca9a8d31f1a7d4d95b5e0887",
        "2b45ba1ba436b9d4ff27f1d3511224c9"
    ]

    def __init__(self, path, **kwargs):
        super().__init__(**kwargs)
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

    def load(self):
        """load"""
        txt_files = []
        for url, md5 in zip(self.urls, self.md5s):
            save_file = f"wn18rr_{os.path.basename(url)}"
            txt_file = util.download(url, self.path, save_file=save_file, md5=md5)
            txt_files.append(txt_file)
        self.load_file(txt_files)
        return self
