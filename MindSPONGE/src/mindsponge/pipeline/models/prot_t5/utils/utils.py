# Copyright 2024 Huawei Technologies Co., Ltd
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
"""some util functions"""
import os
import datetime
import re


def generate_checkpoint_filename(checkpoint_dir, model_info):
    """get datatime of now to generate filename."""
    now = datetime.datetime.now()
    timestamp = now.strftime('%Y%m%d_%H%M%S')
    filename = f'model_{model_info}_{timestamp}.ckpt'
    filepath = os.path.join(checkpoint_dir, filename)
    return filepath


def seqs_tokenizer(sequences, tokenizer, return_tensors=None):
    """tokenizer; data preprocess; UZOB is rare which are replaced in ProtT5 model"""
    sequences = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences]
    tokens = tokenizer(sequences, padding=True, add_special_tokens=True, return_tensors=return_tensors)
    return (tokens['input_ids'], tokens["attention_mask"])
