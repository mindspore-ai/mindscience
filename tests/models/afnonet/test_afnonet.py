# ============================================================================
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
"""AFNONet Test Case"""
import os
import random
import sys
import json

import pytest
import numpy as np

import mindspore as ms
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore import Tensor, ops, set_seed
from mindspore import dtype as mstype

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# pylint: disable=wrong-import-order,wrong-import-position
from tests.tools import compare_output, FP16_RTOL, FP16_ATOL
from mindscience.models.neural_operator.afnonet import AFNONet
from mindscience import RelativeRMSELoss
# pylint: enable=wrong-import-order,wrong-import-position

set_seed(0)
np.random.seed(0)
random.seed(0)

test_data_path = "/home/workspace/mindspore_dataset/mindscience/afnonet"


def _has_files(*paths):
    """Return True only if all given paths exist."""
    return all(os.path.exists(p) for p in paths)


def _load_config_or_default():
    """
    Load AFNONet config strictly from JSON file.

    The file 'afnonet_config.json' must exist in `test_data_path` and contain:
        {
            "image_size": [H, W],
            "in_channels": 1,
            "out_channels": 1,
            "patch_size": 8,
            "encoder_depths": 12,
            "encoder_embed_dim": 768,
            "mlp_ratio": 4,
            "dropout_rate": 1.0,
            "compute_dtype": "float32"
        }
    """
    conf_file = os.path.join(test_data_path, "afnonet_config.json")
    if not os.path.exists(conf_file):
        raise FileNotFoundError(
            f"Config file 'afnonet_config.json' not found in '{test_data_path}'"
        )

    with open(conf_file, "r", encoding="utf-8") as f:
        raw_cfg = json.load(f)

    required_keys = [
        "image_size",
        "in_channels",
        "out_channels",
        "patch_size",
        "encoder_depths",
        "encoder_embed_dim",
        "mlp_ratio",
        "dropout_rate",
        "compute_dtype",
    ]
    missing = [k for k in required_keys if k not in raw_cfg]
    if missing:
        raise KeyError(
            f"Missing keys in afnonet_config.json: {missing}. "
            f"Expected keys: {required_keys}."
        )

    dtype_map = {
        "float32": mstype.float32,
        "float16": mstype.float16,
        "float64": mstype.float64,
        "bf16": mstype.bfloat16,
    }
    compute_dtype_str = str(raw_cfg["compute_dtype"]).lower()
    if compute_dtype_str not in dtype_map:
        raise ValueError(
            f"Unsupported compute_dtype='{raw_cfg['compute_dtype']}' in afnonet_config.json, "
            f"supported: {list(dtype_map.keys())}."
        )

    cfg = {
        "image_size": tuple(raw_cfg["image_size"]),
        "in_channels": int(raw_cfg["in_channels"]),
        "out_channels": int(raw_cfg["out_channels"]),
        "patch_size": int(raw_cfg["patch_size"]),
        "encoder_depths": int(raw_cfg["encoder_depths"]),
        "encoder_embed_dim": int(raw_cfg["encoder_embed_dim"]),
        "mlp_ratio": int(raw_cfg["mlp_ratio"]),
        "dropout_rate": float(raw_cfg["dropout_rate"]),
        "compute_dtype": dtype_map[compute_dtype_str],
    }
    return cfg


def _build_model_from_cfg():
    """
    Build an AFNONet instance according to the loaded configuration.

    Returns:
        AFNONet: Model instance initialized from configuration.
    """
    cfg = _load_config_or_default()
    model = AFNONet(
        image_size=cfg["image_size"],
        in_channels=cfg["in_channels"],
        out_channels=cfg["out_channels"],
        patch_size=cfg["patch_size"],
        encoder_depths=cfg["encoder_depths"],
        encoder_embed_dim=cfg["encoder_embed_dim"],
        mlp_ratio=cfg["mlp_ratio"],
        dropout_rate=cfg["dropout_rate"],
        compute_dtype=cfg["compute_dtype"],
    )
    return model


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_afnonet_forward_accuracy():
    """
    Feature: AFNONet forward accuracy test
    Description: Test the forward accuracy of the AFNONet model in GRAPH_MODE.
    Expectation: The output should match the target prediction data within the specified relative and
                 absolute tolerance values.
    """
    ms.set_device(device_target="Ascend")
    ms.set_context(mode=ms.GRAPH_MODE)

    ckpt_path = os.path.join(test_data_path, "afnonet.ckpt")
    input_path = os.path.join(test_data_path, "afnonet_input.npy")
    output_npz_path = os.path.join(test_data_path, "afnonet_output.npz")

    if not _has_files(ckpt_path, input_path, output_npz_path):
        pytest.skip(
            f"Missing test assets for AFNONet forward: "
            f"{[ckpt_path, input_path, output_npz_path]}"
        )

    model = _build_model_from_cfg()
    params = load_checkpoint(ckpt_path)
    load_param_into_net(model, params)

    input_data = np.load(input_path)
    test_inputs = Tensor(input_data, mstype.float32)

    output = model(test_inputs).asnumpy()

    with np.load(output_npz_path) as data:
        output_tgt = data["output"]

    assert output.shape == output_tgt.shape, (
        f"Shape mismatch: got {output.shape}, expect {output_tgt.shape}"
    )

    validate_ans = compare_output(output, output_tgt, rtol=FP16_RTOL, atol=FP16_ATOL)
    assert validate_ans, "AFNONet forward accuracy verification failed."


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_afnonet_grad_accuracy():
    """
    Feature: AFNONet gradient accuracy test
    Description: Test the accuracy of the computed gradients for the AFNONet model.
    Expectation: The computed gradients should match the reference gradients within the specified relative and
                 absolute tolerance values.
    """
    ms.set_device(device_target="Ascend")
    ms.set_context(mode=ms.GRAPH_MODE)

    ckpt_path = os.path.join(test_data_path, "afnonet.ckpt")
    input_path = os.path.join(test_data_path, "afnonet_input.npy")
    label_path = os.path.join(test_data_path, "afnonet_label.npy")
    grads_npz_path = os.path.join(test_data_path, "afnonet_grads.npz")

    if not _has_files(ckpt_path, input_path, label_path, grads_npz_path):
        pytest.skip(
            f"Missing test assets for AFNONet gradients: "
            f"{[ckpt_path, input_path, label_path, grads_npz_path]}"
        )

    model = _build_model_from_cfg()
    params = load_checkpoint(ckpt_path)
    load_param_into_net(model, params)

    input_data = np.load(input_path)
    label_data = np.load(label_path)
    test_inputs = Tensor(input_data, mstype.float32)
    test_label = Tensor(label_data, mstype.float32)

    loss_func = RelativeRMSELoss()

    def forward_fn(data, label):
        out = model(data)
        loss = loss_func(out, label)
        return loss

    grad_fn = ops.value_and_grad(
        forward_fn, None, model.trainable_params(), has_aux=False
    )
    _, grads = grad_fn(test_inputs, test_label)

    grads_np = tuple(g.asnumpy() for g in grads)

    with np.load(grads_npz_path) as data:
        target_np = tuple(data[key] for key in data.files)

    validate_ans = compare_output(grads_np, target_np, rtol=FP16_RTOL, atol=FP16_ATOL)
    assert validate_ans, "AFNONet gradient accuracy verification failed."


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_afnonet_smoke_shapes_and_dtypes():
    """
    Feature: AFNONet smoke test
    Description: Basic shape/dtype/forward/backward validation without assets.
    Expectation: Forward produces expected shape and dtype; backward produces gradients.
    """
    ms.set_device(device_target="Ascend")
    ms.set_context(mode=ms.GRAPH_MODE)

    cfg = _load_config_or_default()
    batch_size, in_channels = 2, cfg["in_channels"]
    height, width = cfg["image_size"]

    x = Tensor(
        np.random.randn(batch_size, in_channels, height, width).astype(np.float32)
    )

    model = _build_model_from_cfg()

    out = model(x)
    assert out.dtype == mstype.float32
    assert out.ndim == 3

    y = Tensor(np.random.randn(*out.shape).astype(np.float32))
    loss_func = RelativeRMSELoss()

    def forward_fn(data, label):
        o = model(data)
        return loss_func(o, label)

    grad_fn = ops.value_and_grad(forward_fn, None, model.trainable_params())
    loss, grads = grad_fn(x, y)

    assert loss.asnumpy() >= 0.0
    assert any(g is not None for g in grads), "No gradients produced in AFNONet smoke test."
    