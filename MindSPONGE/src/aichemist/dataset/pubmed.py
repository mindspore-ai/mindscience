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
Complex
"""

import os
import re
import csv
import ast
from tqdm import tqdm
from .. import data
from .. import util


class PubMed(data.KnowledgeNodeSet):
    """
    A citation network of scientific publications with TF-IDF word features.

    Statistics:
        - #Node: 19,717
        - #Edge: 44,338
        - #Class: 3

    Parameters:
        path (str): path to store the dataset
        verbose (int, optional): output verbose level
    """

    url = "https://linqs-data.soe.ucsc.edu/public/Pubmed-Diabetes.tgz"
    md5 = "9fa24b917990c47e264a94079b9599fe"

    def __init__(self, path, batch_size=128, verbose=1):
        super().__init__(batch_size=batch_size)
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        zip_file = util.download(self.url, path, md5=self.md5)
        node_file = util.extract(zip_file, "Pubmed-Diabetes/data/Pubmed-Diabetes.NODE.paper.tab")
        edge_file = util.extract(zip_file, "Pubmed-Diabetes/data/Pubmed-Diabetes.DIRECTED.cites.tab")

        inv_node_voc = {}
        node_feature = []
        node_label = []

        with open(node_file, "r", encoding='utf-8') as fin:
            reader = csv.reader(fin, delimiter="\t")
            if verbose:
                reader = iter(tqdm(reader, f"Loading {node_file}", util.get_line_count(node_file)))
            _ = next(reader)
            fields = next(reader)
            group, = re.match(r"cat=(\S+):label", fields[0]).groups()
            label_tokens = group.split(",")
            inv_label_voc = {token: i for i, token in enumerate(label_tokens)}
            inv_feature_voc = {}
            for field in fields[1:]:
                match = re.match(r"numeric:(\S+):0\.0", field)
                if not match:
                    continue
                feature_token, = match.groups()
                inv_feature_voc[feature_token] = len(inv_feature_voc)

            for tokens in reader:
                node_token = tokens[0]
                label_token, = re.match(r"label=(\S+)", tokens[1]).groups()
                feature = [0] * len(inv_feature_voc)
                inv_node_voc[node_token] = len(inv_node_voc)
                for token in tokens[2:]:
                    match = re.match(r"(\S+)=([0-9.]+)", token)
                    if not match:
                        continue
                    feature_token, value = match.groups()
                    key = inv_feature_voc.get(feature_token)
                    feature[key] = ast.literal_eval(value)
                label = inv_label_voc[label_token]
                node_feature.append(feature)
                node_label.append(label)

        edge_list = []

        with open(edge_file, "r", encoding='utf-8') as fin:
            reader = csv.reader(fin, delimiter="\t")
            if verbose:
                reader = iter(tqdm(reader, f"Loading {edge_file}", util.get_line_count(edge_file)))
            _ = next(reader)
            _ = next(reader)
            for tokens in reader:
                h_token, = re.match(r"paper:(\S+)", tokens[1]).groups()
                t_token, = re.match(r"paper:(\S+)", tokens[3]).groups()
                if h_token not in inv_node_voc:
                    inv_node_voc[h_token] = len(inv_node_voc)
                h = inv_node_voc[h_token]
                if t_token not in inv_node_voc:
                    inv_node_voc[t_token] = len(inv_node_voc)
                t = inv_node_voc[t_token]
                edge_list.append((h, t))

        self.load_edge(edge_list, node_feature, node_label, inv_node_voc=inv_node_voc,
                       inv_label_voc=inv_label_voc)
