# ============================================================================
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
"""test mindchemistry ORB"""

import os
import sys
from typing import Optional
import pickle

import requests
import pytest
import numpy as np
import mindspore
from mindspore import nn, Tensor, load_checkpoint, load_param_into_net

from mindchemistry.cell import (
    AttentionInteractionNetwork,
    MoleculeGNS,
    NodeHead,
    GraphHead,
    EnergyHead,
    Orb,
)
import base
from utils import numpy_to_tensor, tensor_to_numpy, is_equal

# pylint: disable=C0413
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(PROJECT_ROOT)
from common.cell import compare_output


def load_graph_data(pkl_path: str):
    """Load graph data from pickle file.
    Args:
        pkl_path: Path to the pickle file
    Returns:
        tuple: (input_graph_ms, output_graph_np)
    """
    with open(pkl_path, "rb") as f:
        loaded = pickle.load(f)

    input_graph_np = loaded["input_graph"]
    output_graph_np = loaded["output_graph"]

    input_graph_ms = base.AtomGraphs(
        *[numpy_to_tensor(getattr(input_graph_np, field))
          for field in input_graph_np._fields]
    )

    return input_graph_ms, output_graph_np


def get_gns(
        latent_dim: int = 256,
        mlp_hidden_dim: int = 512,
        num_message_passing_steps: int = 15,
        num_edge_in_features: int = 23,
        distance_cutoff: bool = True,
        attention_gate: str = "sigmoid",
) -> MoleculeGNS:
    """Define the base pretrained model architecture.
    Args:
        latent_dim: The latent dimension of the model.
        mlp_hidden_dim: The hidden dimension of the MLP layers.
        num_message_passing_steps: The number of message passing steps.
        num_edge_in_features: The number of edge input features.
        distance_cutoff: Whether to use distance cutoff in the interaction.
        attention_gate: The type of attention gate to use.
    Returns:
        MoleculeGNS: The MoleculeGNS model instance.
    """
    return MoleculeGNS(
        num_node_in_features=256,
        num_node_out_features=3,
        num_edge_in_features=num_edge_in_features,
        latent_dim=latent_dim,
        interactions="simple_attention",
        interaction_params={
            "distance_cutoff": distance_cutoff,
            "polynomial_order": 4,
            "cutoff_rmax": 6,
            "attention_gate": attention_gate,
        },
        num_message_passing_steps=num_message_passing_steps,
        num_mlp_layers=2,
        mlp_hidden_dim=mlp_hidden_dim,
        use_embedding=True,
        node_feature_names=["feat"],
        edge_feature_names=["feat"],
    )


def load_model_for_inference(model: nn.Cell, weights_path: str) -> nn.Cell:
    """Load a pretrained model in inference mode.
    Args:
        model: The model to load the weights into.
        weights_path: Path to the checkpoint file.
    Returns:
        nn.Cell: The model with loaded weights.
    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        ValueError: If the checkpoint file has more parameters than the model.
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Checkpoint file {weights_path} not found.")
    param_dict = load_checkpoint(weights_path)

    try:
        load_param_into_net(model, param_dict)
    except ValueError:
        print("Warning: The checkpoint file has more parameters than the model. \
              This may be due to a mismatch in the model architecture or version.")
        params = []
        for key in param_dict:
            params.append(param_dict[key])
        for parameters in model.trainable_params():
            param_ckpt = params.pop(0)
            assert parameters.shape == param_ckpt.shape, f"Shape mismatch: {parameters.name}"
            param_ckpt = param_ckpt.reshape(parameters.shape)
            parameters.set_data(param_ckpt)

    model.set_train(False)
    return model

def orb_v2(weights_path: Optional[str]) -> nn.Cell:
    """Load ORB v2.
    Args:
        weights_path: Path to the checkpoint file.
    Returns:
        Orb GraphRegressor: The ORB v2 model instance.
    """
    gns = get_gns()

    model = Orb(
        graph_head=EnergyHead(
            latent_dim=256,
            num_mlp_layers=1,
            mlp_hidden_dim=256,
            target_property_dim=1,
            node_aggregation="mean",
            reference_energy_name="vasp-shifted",
            train_reference=True,
            predict_atom_avg=True,
        ),
        node_head=NodeHead(
            latent_dim=256,
            num_mlp_layers=1,
            mlp_hidden_dim=256,
            target_property_dim=3,
            remove_mean=True,
        ),
        stress_head=GraphHead(
            latent_dim=256,
            num_mlp_layers=1,
            mlp_hidden_dim=256,
            target_property_dim=6,
            compute_stress=True,
        ),
        model=gns,
    )
    model = load_model_for_inference(model, weights_path)
    return model


def orb_mptraj_only_v2(
        weights_path: Optional[str] = None,
):
    """Load ORB MPTraj Only v2."""

    return orb_v2(weights_path,)


def download_file(url, local_filename):
    """Download a file from a URL to a local path."""
    response = requests.get(url, timeout=30)
    if response.status_code == 200:
        with open(local_filename, 'wb') as f:
            f.write(response.content)
    else:
        print(f"Failed to download file. HTTP Status Code: {response.status_code}")


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_attn():
    """
    Feature: Test AttentionInteractionNetwork in platform ascend.
    Description: The forward output should has expected shape and accuracy.
    Expectation: Success or throw AssertionError.
    """
    mindspore.set_context(mode=mindspore.PYNATIVE_MODE)
    # prepare data
    download_file(
        'https://download-mindspore.osinfra.cn/mindscience/mindchemistry/orb/test/attn_input_output.pkl',
        'attn_input_output.pkl'
    )
    download_file(
        'https://download-mindspore.osinfra.cn/mindscience/mindchemistry/orb/orb_ckpts/attn_net.ckpt',
        'attn_net.ckpt'
    )
    input_graph_ms, output_graph_np = load_graph_data('attn_input_output.pkl')

    attn_net = AttentionInteractionNetwork(
        num_node_in=256,
        num_node_out=256,
        num_edge_in=256,
        num_edge_out=256,
        num_mlp_layers=2,
        mlp_hidden_dim=512,
    )

    # load checkpoint
    param_dict = load_checkpoint('attn_net.ckpt')
    load_param_into_net(attn_net, param_dict)

    # inference
    edges, nodes = attn_net(
        input_graph_ms.edge_features,
        input_graph_ms.node_features,
        input_graph_ms.senders,
        input_graph_ms.receivers,
    )

    # Validate results
    out_node_feats = tensor_to_numpy(nodes["feat"])
    out_edge_feats = tensor_to_numpy(edges["feat"])
    out_node_feats_np = output_graph_np.node_features["feat"]
    out_edge_feats_np = output_graph_np.edge_features["feat"]

    flag_node = is_equal(out_node_feats, out_node_feats_np)
    flag_edge = is_equal(out_edge_feats, out_edge_feats_np)
    assert flag_node, "Failed! Node features mismatch in attention network"
    assert flag_edge, "Failed! Edge features mismatch in attention network"


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_gns():
    """
    Feature: Test MoleculeGNS network in platform ascend.
    Description: The forward output should has expected shape and accuracy.
    Expectation: Success or throw AssertionError.
    """
    mindspore.set_context(mode=mindspore.PYNATIVE_MODE)
    download_file(
        'https://download-mindspore.osinfra.cn/mindscience/mindchemistry/orb/test/gns_input_output.pkl',
        'gns_input_output.pkl'
    )
    download_file(
        'https://download-mindspore.osinfra.cn/mindscience/mindchemistry/orb/orb_ckpts/gns_net.ckpt',
        'gns_net.ckpt'
    )
    input_graph_ms, output_graph_np = load_graph_data('gns_input_output.pkl')

    # load gns model and checkpoint
    gns_model = get_gns()

    # load checkpoint
    param_dict = load_checkpoint('gns_net.ckpt')
    load_param_into_net(gns_model, param_dict)

    edges, nodes = gns_model(
        input_graph_ms.edge_features,
        input_graph_ms.node_features,
        input_graph_ms.senders,
        input_graph_ms.receivers,
    )

    out_node_feats = tensor_to_numpy(nodes["feat"])
    out_edge_feats = tensor_to_numpy(edges["feat"])
    out_node_feats_np = output_graph_np.node_features["feat"]
    out_edge_feats_np = output_graph_np.edge_features["feat"]

    flag_node = is_equal(out_node_feats, out_node_feats_np)
    flag_edge = is_equal(out_edge_feats, out_edge_feats_np)
    assert flag_node, "Failed! Node features mismatch in MoleculeGNS network"
    assert flag_edge, "Failed! Edge features mismatch in MoleculeGNS network"


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_node_head():
    """
    Feature: Test NodeHead in platform ascend.
    Description: The forward output should has expected shape and accuracy.
    Expectation: Success or throw AssertionError.
    """
    mindspore.set_context(mode=mindspore.PYNATIVE_MODE)
    node_head = NodeHead(
        latent_dim=256,
        num_mlp_layers=1,
        mlp_hidden_dim=256,
        target_property_dim=3,
        remove_mean=True,
    )

    n_atoms = 4
    n_node = Tensor([n_atoms], mindspore.int32)
    atomic_numbers = Tensor(np.random.randint(1, 119, size=(n_atoms,), dtype=np.int32))
    atomic_numbers_embedding_np = np.zeros((n_atoms, 118), dtype=np.float32)
    for i, num in enumerate(atomic_numbers.asnumpy()):
        atomic_numbers_embedding_np[i, num - 1] = 1.0

    node_features = {
        "atomic_numbers": atomic_numbers,
        "atomic_numbers_embedding": Tensor(atomic_numbers_embedding_np),
        "positions": Tensor(np.random.randn(n_atoms, 3).astype(np.float32)),
        "feat": Tensor(np.random.randn(n_atoms, 256).astype(np.float32))
    }

    output = node_head(node_features, n_node)
    assert output['node_pred'].shape == (4, 3), \
        f"Expected node_pred shape (4, 3), but got {output['node_pred'].shape}"


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_graph_head():
    """
    Feature: Test GraphHead in platform ascend.
    Description: The forward output should has expected shape and accuracy.
    Expectation: Success or throw AssertionError.
    """
    mindspore.set_context(mode=mindspore.PYNATIVE_MODE)
    graph_head = GraphHead(
        latent_dim=256,
        num_mlp_layers=1,
        mlp_hidden_dim=256,
        target_property_dim=6,
        compute_stress=True,
    )

    n_atoms = 4
    n_node = Tensor([n_atoms], mindspore.int32)
    atomic_numbers = Tensor(np.random.randint(1, 119, size=(n_atoms,), dtype=np.int32))
    atomic_numbers_embedding_np = np.zeros((n_atoms, 118), dtype=np.float32)
    for i, num in enumerate(atomic_numbers.asnumpy()):
        atomic_numbers_embedding_np[i, num - 1] = 1.0

    node_features = {
        "atomic_numbers": atomic_numbers,
        "atomic_numbers_embedding": Tensor(atomic_numbers_embedding_np),
        "positions": Tensor(np.random.randn(n_atoms, 3).astype(np.float32)),
        "feat": Tensor(np.random.randn(n_atoms, 256).astype(np.float32))
    }

    output = graph_head(node_features, n_node)
    assert output['stress_pred'].shape == (1, 6), \
        f"Expected stress_pred shape (1, 6), but got {output['stress_pred'].shape}"


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_energy_head():
    """
    Feature: Test EnergyHead in platform ascend.
    Description: The forward output should has expected shape and accuracy.
    Expectation: Success or throw AssertionError.
    """
    mindspore.set_context(mode=mindspore.PYNATIVE_MODE)
    energy_head = EnergyHead(
        latent_dim=256,
        num_mlp_layers=1,
        mlp_hidden_dim=256,
        target_property_dim=1,
        node_aggregation="mean",
        reference_energy_name="vasp-shifted",
        train_reference=True,
        predict_atom_avg=True,
    )

    n_atoms = 4
    n_node = Tensor([n_atoms], mindspore.int32)
    atomic_numbers = Tensor(np.random.randint(1, 119, size=(n_atoms,), dtype=np.int32))
    atomic_numbers_embedding_np = np.zeros((n_atoms, 118), dtype=np.float32)
    for i, num in enumerate(atomic_numbers.asnumpy()):
        atomic_numbers_embedding_np[i, num - 1] = 1.0

    node_features = {
        "atomic_numbers": atomic_numbers,
        "atomic_numbers_embedding": Tensor(atomic_numbers_embedding_np),
        "positions": Tensor(np.random.randn(n_atoms, 3).astype(np.float32)),
        "feat": Tensor(np.random.randn(n_atoms, 256).astype(np.float32))
    }

    output = energy_head(node_features, n_node)
    assert output['graph_pred'].shape == (1, 1), \
        f"Expected graph_pred shape {(1, 1)}, but got {output['graph_pred'].shape}"


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_inference():
    """
    Feature: Test Orb network in platform ascend.
    Description: The forward output should has expected shape and accuracy.
    Expectation: Success or throw AssertionError.
    """
    mindspore.set_context(mode=mindspore.PYNATIVE_MODE)
    download_file(
        'https://download-mindspore.osinfra.cn/mindscience/mindchemistry/orb/test/orb_input_output.pkl',
        'orb_input_output.pkl'
    )
    download_file(
        'https://download-mindspore.osinfra.cn/mindscience/mindchemistry/orb/orb_ckpts/orb-mptraj-only-v2.ckpt',
        'orb-mptraj-only-v2.ckpt'
    )
    reference_path = 'orb_input_output.pkl'
    with open(reference_path, "rb") as f:
        loaded = pickle.load(f)

    atom_graph_ms = loaded["input_graph"]
    output_pt = loaded["output"]

    regressor = orb_mptraj_only_v2(weights_path='orb-mptraj-only-v2.ckpt')
    regressor.set_train(False)

    out_ms = regressor.predict(
        atom_graph_ms.edge_features,
        atom_graph_ms.node_features,
        atom_graph_ms.senders,
        atom_graph_ms.receivers,
        atom_graph_ms.n_node,
        atom_graph_ms.atomic_numbers,
    )

    out_ms = {k: tensor_to_numpy(v) for k, v in out_ms.items()}

    for k in out_ms:
        flag = compare_output(out_ms[k], output_pt[k])
        assert flag, f"Failed! Orb network inference output {k} mismatch"
