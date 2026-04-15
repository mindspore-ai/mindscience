# Copyright 2025 Huawei Technologies Co., Ltd & CPL YiQin GAO Research Group
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

"""init"""
import warnings
from pathlib import Path

# .pkl files download from https://huggingface.co/ctheodoris/Geneformer/tree/main/geneformer

GENE_MEDIAN_FILE = Path(__file__).parent / "gene_median_dictionary_gc95M.pkl"
TOKEN_DICTIONARY_FILE = Path(__file__).parent / "token_dictionary_gc95M.pkl"
ENSEMBL_DICTIONARY_FILE = Path(__file__).parent / "gene_name_id_dict_gc95M.pkl"
ENSEMBL_MAPPING_FILE = Path(__file__).parent / "ensembl_mapping_dict_gc95M.pkl"
