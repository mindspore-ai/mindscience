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

#include <math.h>
#include <stdio.h>
#include <vector>
#include "./neighbor_list.h"

static void Find_Atom_Neighbors(const int atom_numbers, int max_neighbor_numbers, int max_atom_in_grid_numbers,
                                int kGridSize, const float *cutoff_skin_square, const int *atom_in_grid_serial,
                                const VECTOR *crd, const float *box_length, const int *gpointer, const int *bucket,
                                const int *atom_numbers_in_grid_bucket, int *nl_atom_numbers, int *nl_atom_serial) {
  int atom_j;
  int grid_serial2;
  float x, y, z;
  VECTOR dr;
  float dr2;

  for (int atom_i = 0; atom_i < atom_numbers; atom_i++) {
    int grid_serial = atom_in_grid_serial[atom_i];
    int atom_numbers_in_nl_lin = 0;
    VECTOR crd_i = crd[atom_i];
    for (int grid_cycle = 0; grid_cycle < 125; grid_cycle = grid_cycle + 1) {
      grid_serial2 = gpointer[grid_serial * kGridSize + grid_cycle];
      for (int i = 0; i < atom_numbers_in_grid_bucket[grid_serial2]; i = i + 1) {
         atom_j = bucket[grid_serial2 * max_atom_in_grid_numbers + i];
        if (atom_j > atom_i) {
          x = crd[atom_j].x - crd_i.x;
          y = crd[atom_j].y - crd_i.y;
          z = crd[atom_j].z - crd_i.z;
          dr.x = x - floorf(x / box_length[0] + 0.5) * box_length[0];
          dr.y = y - floorf(y / box_length[1] + 0.5) * box_length[1];
          dr.z = z - floorf(z / box_length[2] + 0.5) * box_length[2];
          dr2 = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;
          if (dr2 < *cutoff_skin_square) {
            nl_atom_serial[atom_i * max_neighbor_numbers + atom_numbers_in_nl_lin] = atom_j;
            atom_numbers_in_nl_lin = atom_numbers_in_nl_lin + 1;
          }
        }
      }
    }
    nl_atom_numbers[atom_i] = atom_numbers_in_nl_lin;
  }
}

static void Copy_Neighbor_List_Atom_Number(int atom_numbers, int max_neighbor_numbers,
                                           int *nl_atom_numbers, int *nl_atom_serial,
                                           int *output_nl_atom_numbers, int *output_nl_atom_serial) {
  for (int i = 0; i < atom_numbers; i++) {
    for (int j = 0; j < max_neighbor_numbers; j++) {
      output_nl_atom_serial[i * max_neighbor_numbers + j] = nl_atom_serial[i * max_neighbor_numbers + j];
    }
    output_nl_atom_numbers[i] = nl_atom_numbers[i];
  }
}

const int max_neighbor_numbers = 800;
const int max_atom_in_grid_numbers = 64;
const int kGridSize = 125;

extern "C" int FindAtomNeighbors(int nparam, void **params, int *ndims, int64_t **shapes,
                                 const char **dtypes, void *stream, void *extra) {
  int atom_number;

  float *cutoff_square = (static_cast<float *>(params[0]));
  int *atom_in_grid_serial = static_cast<int *>(params[1]);
  float *coordinate = static_cast<float *>(params[2]);
  float *box_length = static_cast<float *>(params[3]);
  int *gpointer = static_cast<int *>(params[4]);
  int *bucket = static_cast<int *>(params[5]);
  int *atom_numbers_in_grid_bucket = static_cast<int *>(params[6]);
  int *nl_atom_numbers =  static_cast<int *>(params[7]);
  int *nl_atom_serial = static_cast<int *>(params[8]);

  int *output_nl_atom_numbers =  static_cast<int *>(params[9]);
  int *output_nl_atom_serial = static_cast<int *>(params[10]);

  atom_number = shapes[1][0];
  Find_Atom_Neighbors(atom_number, max_neighbor_numbers, max_atom_in_grid_numbers,
    kGridSize, cutoff_square, atom_in_grid_serial, reinterpret_cast<VECTOR *>(coordinate),
    box_length, gpointer, bucket, atom_numbers_in_grid_bucket, nl_atom_numbers, nl_atom_serial);
  Copy_Neighbor_List_Atom_Number(atom_number, max_neighbor_numbers, nl_atom_numbers,
    nl_atom_serial, output_nl_atom_numbers, output_nl_atom_serial);
  return 0;
}
