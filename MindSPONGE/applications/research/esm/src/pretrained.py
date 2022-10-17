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
"""Load pretraining model"""

from argparse import Namespace
import json
from pathlib import Path
from gvp_transformer import GVPTransformerModel
from mindspore import load_checkpoint, load_param_into_net
from mindspore import context
context.set_context(mode=context.GRAPH_MODE, device_target="GPU")



def load_model_and_alphabet_local(model_location):
    """ Load from local path. The regression weights need to be co-located """
    model_location = Path(model_location)
    model_data = load_checkpoint(str(model_location))
    regression_data = None
    return load_model_and_alphabet_core(model_data, regression_data)


def load_model_and_alphabet_core(model_data, regression_data=None):
    """ Load model and alphabet"""
    import data  # conditional esm.inverse_folding below
    if regression_data is not None:
        model_data["model"].update(regression_data["model"])

    alphabet = data.Alphabet.from_architecture('vt_medium_with_invariant_gvp')

    with open('src/args.json', 'r') as args_data:
        model_args = json.load(args_data)
    model = GVPTransformerModel(Namespace(**model_args), alphabet,)
    load_param_into_net(model, model_data)

    return model, alphabet


def esm_if1_gvp4_t16_142m_ur50():
    """Load esm_if1_gvp4_t16_142M_UR50.ckpt"""
    return load_model_and_alphabet_local('esm_if1_gvp4_t16_142M_UR50.ckpt')
