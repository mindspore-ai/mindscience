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
"""train model using jarvis dataset"""
from typing import Any, Dict, Union, Optional
from pydantic import BaseSettings as PydanticBaseSettings
from pydantic.typing import Literal
from data.data import get_train_val_loaders


class BaseSettings(PydanticBaseSettings):
    """Add configuration to default Pydantic BaseSettings."""

    class Config:
        """Configure BaseSettings behavior."""

        extra = "forbid"
        use_enum_values = True
        env_prefix = "jv_"


class TrainingConfig(BaseSettings):
    """Training config defaults and validation."""

    # dataset configuration
    dataset: Literal[
        "dft_3d",
        "megnet",
    ] = "dft_3d"
    target: Literal["formation_energy_peratom", "final_energy"] = "formation_energy_peratom"
    atom_features: Literal["basic", "atomic_number", "cfid", "cgcnn"] = "cgcnn"
    neighbor_strategy: Literal["k-nearest", "voronoi", "pairwise-k-nearest"] = "k-nearest"
    id_tag: Literal["jid", "id", "_oqmd_entry_id"] = "jid"

    # training configuration
    random_seed: Optional[int] = 100
    classification_threshold: Optional[float] = None
    n_val: Optional[int] = None
    n_test: Optional[int] = None
    n_train: Optional[int] = None
    train_ratio: Optional[float] = 0.8
    val_ratio: Optional[float] = 0.1
    test_ratio: Optional[float] = 0.1
    use_canonize: bool = True
    cutoff: float = 8.0
    max_neighbors: int = 12
    keep_data_order: bool = False
    distributed: bool = False
    use_lattice: bool = False
    dataset_path: dict = None


def get_loader(
        config: Union[TrainingConfig, Dict[str, Any]],
):
    """
    `config` should conform to matformer.conf.TrainingConfig, and
    if passed as a dict with matching keys, pydantic validation is used
    """
    config = TrainingConfig(**config)
    line_graph = True
    get_train_val_loaders(
        dataset=config.dataset,
        target=config.target,
        n_train=config.n_train,
        n_val=config.n_val,
        n_test=config.n_test,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        atom_features=config.atom_features,
        line_graph=line_graph,
        id_tag=config.id_tag,
        use_canonize=config.use_canonize,
        cutoff=config.cutoff,
        max_neighbors=config.max_neighbors,
        classification_threshold=config.classification_threshold,
        keep_data_order=config.keep_data_order,
        use_lattice=config.use_lattice,
        dataset_path=config.dataset_path
    )


def get_prop_model(
        prop="",
        dataset="dft_3d",
        classification_threshold=None,
        use_lattice=False,
        neighbor_strategy="k-nearest",
        atom_features="cgcnn",
        dataset_path=None
):
    """Train models for a dataset and a property."""
    config = {
        "dataset": dataset,
        "target": prop,
        "classification_threshold": classification_threshold,
        "atom_features": atom_features,
    }
    config["use_lattice"] = use_lattice
    config["neighbor_strategy"] = neighbor_strategy
    config["dataset_path"] = dataset_path

    get_loader(config)
