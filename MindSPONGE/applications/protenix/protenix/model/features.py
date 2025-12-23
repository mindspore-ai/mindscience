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
"""Data-side of the input features processing."""

import dataclasses
from typing_extensions import Any, Self
import mindspore as ms


@dataclasses.dataclass
class MSA:
    """Dataclass containing MSA."""
    rows: Any = None
    mask: Any = None
    deletion_matrix: Any = None

    def index_msa_rows(self, indices) -> Self:
        if indices.ndim != 1:
            raise ValueError(f"Indices dimension {indices.ndim} is not 1")

        return MSA(
            rows=self.rows[indices, :],
            mask=self.mask[indices, :],
            deletion_matrix=self.deletion_matrix[indices, :],
        )


@dataclasses.dataclass
class TokenFeatures:
    """Dataclass containing features for tokens."""
    residue_index: Any = None
    token_index: Any = None
    aatype: Any = None
    mask: Any = None
    seq_length: Any = None


@dataclasses.dataclass
class RefStructure:
    """Contains ref structure information."""
    positions: Any = None
    mask: Any = None
    element: Any = None
    charge: Any = None
    atom_name_chars: Any = None
    ref_space_uid: Any = None


@dataclasses.dataclass
class AtomCrossAtt:
    """Operate on flat atoms."""
    token_atoms_to_queries: ms.Tensor = None
