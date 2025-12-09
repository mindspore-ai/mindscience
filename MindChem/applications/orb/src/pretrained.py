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
"""pretrained."""

import os
from typing import Optional

from mindspore import nn, load_checkpoint, load_param_into_net

from models import (
    EnergyHead,
    GraphHead,
    Orb,
    NodeHead,
    MoleculeGNS,
)


def get_gns(
        latent_dim: int = 256,
        mlp_hidden_dim: int = 512,
        num_message_passing_steps: int = 15,
        num_edge_in_features: int = 23,
        distance_cutoff: bool = True,
        attention_gate: str = "sigmoid",
) -> MoleculeGNS:
    """Define the base pretrained model architecture."""
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
    """
    Load a pretrained model in inference mode, using GPU if available.
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Checkpoint file {weights_path} not found.")
    param_dict = load_checkpoint(weights_path)
    load_param_into_net(model, param_dict)
    model.set_train(False)

    return model

def orb_v2(
        weights_path: Optional[str] = None,
):
    """Load ORB v2."""
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
