/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * Note:
 *  NeighborListUpdate. This is an experimental interface that is subject to change and/or deletion.
 */

#include "./neighbor_list.cuh"

static __global__ void Delete_Excluded_Atoms(const int atom_numbers, const int max_neighbor_numbers,
                                             const int *excluded_list_start, const int *excluded_list,
                                             const int *excluded_atom_numbers,
                                             int *nl_atom_numbers, int *nl_atom_serial) {
  int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (atom_i < atom_numbers) {
    int excluded_number = excluded_atom_numbers[atom_i];
    if (excluded_number > 0) {
      int list_start = excluded_list_start[atom_i];
      int atom_min = excluded_list[list_start];
      int list_end = list_start + excluded_number;
      int atom_max = excluded_list[list_end - 1];
      int atomnumbers_in_nl_lin = nl_atom_numbers[atom_i];
      int atom_j;
      int excluded_atom_numbers_lin = list_end - list_start;
      int excluded_atom_numbers_count = 0;
      for (int i = 0; i < atomnumbers_in_nl_lin; i = i + 1) {
        atom_j = nl_atom_serial[atom_i * max_neighbor_numbers + i];
        if (atom_j < atom_min || atom_j > atom_max) {
          continue;
        } else {
          for (int j = list_start; j < list_end; j = j + 1) {
            if (atom_j == excluded_list[j]) {
              atomnumbers_in_nl_lin = atomnumbers_in_nl_lin - 1;
              nl_atom_serial[atom_i * max_neighbor_numbers + i] = nl_atom_serial[atom_i * max_neighbor_numbers +
                atomnumbers_in_nl_lin];
              excluded_atom_numbers_count = excluded_atom_numbers_count + 1;
              i = i - 1;
            }
          }
          if (excluded_atom_numbers_count < excluded_atom_numbers_lin) {
          } else {
            break;
          }
        }
      }
      nl_atom_numbers[atom_i] = atomnumbers_in_nl_lin;
    }
  }
}

static __global__ void Copy_Neighbor_List_Atom_Number(int atom_numbers, int max_neighbor_numbers,
                                                      int *nl_atom_numbers, int *nl_atom_serial,
                                                      int *output_nl_atom_numbers, int *output_nl_atom_serial) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < atom_numbers) {
    for (int j = 0; j < max_neighbor_numbers; j++) {
      output_nl_atom_serial[i * max_neighbor_numbers + j] = nl_atom_serial[i * max_neighbor_numbers + j];
    }
    output_nl_atom_numbers[i] = nl_atom_numbers[i];
  }
}

const int max_neighbor_numbers = 800;
extern "C" int DeleteExcludedAtoms(int nparam, void **params, int *ndims, int64_t **shapes,
                                   const char **dtypes, void *stream, void *extra) {
  int atom_number;

  int *excluded_list_start = static_cast<int *>(params[0]);
  int *excluded_list = static_cast<int *>(params[1]);
  int *excluded_numbers = static_cast<int *>(params[2]);
  int *nl_atom_numbers = static_cast<int *>(params[3]);
  int *nl_atom_serial = static_cast<int *>(params[4]);

  int *output_nl_atom_numbers = static_cast<int *>(params[5]);
  int *output_nl_atom_serial = static_cast<int *>(params[6]);

  atom_number = shapes[3][0];
  Delete_Excluded_Atoms<<<ceilf(static_cast<float>(atom_number) / 32), 32, 0,
    reinterpret_cast<cudaStream_t>(stream)>>>(atom_number, max_neighbor_numbers, excluded_list_start,
    excluded_list, excluded_numbers, nl_atom_numbers, nl_atom_serial);
  Copy_Neighbor_List_Atom_Number<<<ceilf(static_cast<float>(atom_number) / 128), 128, 0,
    reinterpret_cast<cudaStream_t>(stream)>>>(atom_number, max_neighbor_numbers, nl_atom_numbers,
    nl_atom_serial, output_nl_atom_numbers, output_nl_atom_serial);
  return 0;
}
