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


from functools import lru_cache
from typing import Dict, List

import mindscience.e3nn.o3 as o3
import mindspore as ms
from mindspore import Tensor
from se3_transformer.runtime.utils import degree_to_dim


@lru_cache(maxsize=None)
def get_clebsch_gordon(J: int, d_in: int, d_out: int) -> Tensor:
    """Get the (cached) Q^{d_out,d_in}_J matrices from equation (8)"""
    return o3.wigner_3j(J, d_in, d_out, dtype=ms.float32).permute(2, 1, 0)


@lru_cache(maxsize=None)
def get_all_clebsch_gordon(max_degree: int) -> List[List[Tensor]]:
    all_cb = []
    for d_in in range(max_degree + 1):
        for d_out in range(max_degree + 1):
            K_Js = []
            for J in range(abs(d_in - d_out), d_in + d_out + 1):
                K_Js.append(get_clebsch_gordon(J, d_in, d_out))
            all_cb.append(K_Js)
    return all_cb


def get_spherical_harmonics(relative_pos: Tensor, max_degree: int) -> List[Tensor]:
    all_degrees = list(range(2 * max_degree + 1))
    sh = o3.spherical_harmonics(all_degrees, relative_pos, normalize=True)
    return ms.mint.split(sh, [degree_to_dim(d) for d in all_degrees], dim=1)


def get_basis_script(
    max_degree: int,
    use_pad_trick: bool,
    spherical_harmonics: List[Tensor],
    clebsch_gordon: List[List[Tensor]],
) -> Dict[str, Tensor]:
    """
    Compute pairwise bases matrices for degrees up to max_degree
    :param max_degree:            Maximum input or output degree
    :param use_pad_trick:         Pad some of the odd dimensions for a better use of Tensor Cores
    :param spherical_harmonics:   List of computed spherical harmonics
    :param clebsch_gordon:        List of computed CB-coefficients
    :param amp:                   When true, return bases in FP16 precision
    """
    basis = {}
    idx = 0

    for d_in in range(max_degree + 1):
        for d_out in range(max_degree + 1):
            key = f"{d_in}_{d_out}"
            K_Js = []
            for freq_idx, J in enumerate(range(abs(d_in - d_out), d_in + d_out + 1)):
                Q_J = clebsch_gordon[idx][freq_idx]
                K_Js.append(
                    ms.mint.einsum(
                        "n f, k l f -> n l k",
                        spherical_harmonics[J].float(),
                        Q_J.float(),
                    )
                )

            # Stack on second dim so order is n l f k
            basis[key] = ms.mint.stack(K_Js, 2)

            # Pad the k dimension, that can be sliced later
            if use_pad_trick:
                basis[key] = ms.ops.pad(basis[key], (0, 1))

            idx += 1

    return basis


def update_basis_with_fused(
    basis: Dict[str, Tensor], max_degree: int, use_pad_trick: bool, fully_fused: bool
) -> Dict[str, Tensor]:
    """Update the basis dict with partially and optionally fully fused bases"""
    num_edges = basis["0_0"].shape[0]
    dtype = basis["0_0"].dtype
    sum_dim = sum([degree_to_dim(d) for d in range(max_degree + 1)])

    # Fused per output degree
    for d_out in range(max_degree + 1):
        dout_dim = degree_to_dim(d_out)

        freq_sizes = [degree_to_dim(min(d, d_out)) for d in range(max_degree + 1)]
        sum_freq = sum(freq_sizes)

        rows = []
        acc_f = 0
        for d_in in range(max_degree + 1):
            din_dim = degree_to_dim(d_in)
            fsize = freq_sizes[d_in]

            block = basis[f"{d_in}_{d_out}"][
                :, :, :, :dout_dim
            ]  # (E, din_dim, fsize, dout_dim)

            left = ms.mint.zeros((num_edges, din_dim, acc_f, dout_dim), dtype=dtype)
            right = ms.mint.zeros(
                (num_edges, din_dim, sum_freq - acc_f - fsize, dout_dim), dtype=dtype
            )
            row = ms.mint.cat(
                [left, block, right], dim=2
            )  # (E, din_dim, sum_freq, dout_dim)

            rows.append(row)
            acc_f += fsize

        basis_fused = ms.mint.cat(rows, dim=1)  # (E, sum_dim, sum_freq, dout_dim)

        if use_pad_trick:
            pad = ms.mint.zeros((num_edges, sum_dim, sum_freq, 1), dtype=dtype)
            basis_fused = ms.mint.cat(
                [basis_fused, pad], dim=3
            )  # (E, sum_dim, sum_freq, dout_dim+1)

        basis[f"out{d_out}_fused"] = basis_fused

    # Fused per input degree
    for d_in in range(max_degree + 1):
        din_dim = degree_to_dim(d_in)

        freq_sizes = [
            degree_to_dim(min(d_out, d_in)) for d_out in range(max_degree + 1)
        ]
        sum_freq = sum(freq_sizes)

        cols = []
        acc_f = 0
        for d_out in range(max_degree + 1):
            dout_dim = degree_to_dim(d_out)
            fsize = freq_sizes[d_out]

            block = basis[f"{d_in}_{d_out}"][
                :, :, :, :dout_dim
            ]  # (E, din_dim, fsize, dout_dim)

            left = ms.mint.zeros((num_edges, din_dim, acc_f, dout_dim), dtype=dtype)
            right = ms.mint.zeros(
                (num_edges, din_dim, sum_freq - acc_f - fsize, dout_dim), dtype=dtype
            )
            col = ms.mint.cat(
                [left, block, right], dim=2
            )  # (E, din_dim, sum_freq, dout_dim)

            cols.append(col)
            acc_f += fsize

        basis_fused = ms.mint.cat(cols, dim=3)  # (E, din_dim, sum_freq, sum_dim)
        basis[f"in{d_in}_fused"] = basis_fused

    if fully_fused:
        # Fully fused
        sum_freq = sum(
            [
                sum([degree_to_dim(min(d_in, d_out)) for d_in in range(max_degree + 1)])
                for d_out in range(max_degree + 1)
            ]
        )
        blocks = []
        acc_f, acc_d = 0, 0

        for d_out in range(max_degree + 1):
            b = basis[
                f"out{d_out}_fused"
            ]  # (E, sum_dim, freq_d_out, degree_to_dim(d_out) [+ pad])
            dout_dim = degree_to_dim(d_out)
            freq_d_out = b.shape[2]

            b_core = b[:, :, :, :dout_dim]  # (num_edges, sum_dim, freq_d_out, dout_dim)

            left_f = ms.mint.zeros((num_edges, sum_dim, acc_f, dout_dim), dtype=dtype)
            right_f = ms.mint.zeros(
                (num_edges, sum_dim, sum_freq - acc_f - freq_d_out, dout_dim),
                dtype=dtype,
            )
            b_freq = ms.mint.cat(
                [left_f, b_core, right_f], dim=2
            )  # (E, sum_dim, sum_freq, dout_dim)

            left_d = ms.mint.zeros((num_edges, sum_dim, sum_freq, acc_d), dtype=dtype)
            right_d = ms.mint.zeros(
                (num_edges, sum_dim, sum_freq, sum_dim - acc_d - dout_dim), dtype=dtype
            )
            b_full = ms.mint.cat(
                [left_d, b_freq, right_d], dim=3
            )  # (E, sum_dim, sum_freq, sum_dim)

            blocks.append(b_full)

            acc_f += freq_d_out
            acc_d += dout_dim

        basis_fused = blocks[0]
        for blk in blocks[1:]:
            basis_fused = basis_fused + blk

        basis["fully_fused"] = basis_fused

    # We know that the basis for l = k = 0 is filled with a constant
    del basis["0_0"]
    return basis


def get_basis(
    relative_pos: Tensor, max_degree: int = 4, use_pad_trick: bool = False
) -> Dict[str, Tensor]:

    spherical_harmonics = get_spherical_harmonics(relative_pos, max_degree)
    clebsch_gordon = get_all_clebsch_gordon(max_degree)

    basis = get_basis_script(
        max_degree=max_degree,
        use_pad_trick=use_pad_trick,
        spherical_harmonics=spherical_harmonics,
        clebsch_gordon=clebsch_gordon,
    )
    return basis
