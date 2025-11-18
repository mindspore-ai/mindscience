# Modified from se3-transformer-public (https://github.com/FabianFuchsML/se3-transformer-public)
# Original license: MIT License
#
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

import argparse
import random
from typing import Dict, List, Union

import mindspore as ms
import numpy as np
from mindspore import Tensor, mint


def aggregate_residual(feats1, feats2, method: str):
    """Add or concatenate two fiber features together. If degrees don't match, will use the ones of feats2."""
    if method in ["add", "sum"]:
        return {k: (v + feats1[k]) if k in feats1 else v for k, v in feats2.items()}
    elif method in ["cat", "concat"]:
        return {
            k: mint.cat([v, feats1[k]], dim=1) if k in feats1 else v
            for k, v in feats2.items()
        }
    else:
        raise ValueError("Method must be add/sum or cat/concat")


def degree_to_dim(degree: int) -> int:
    return 2 * degree + 1


def unfuse_features(features: Tensor, degrees: List[int]) -> Dict[str, Tensor]:
    return dict(
        zip(
            map(str, degrees),
            features.split([degree_to_dim(deg) for deg in degrees], dim=-1),
        )
    )


def str2bool(v: Union[bool, str]) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def seed_everything(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    ms.manual_seed(seed)
