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

static __global__ void Clear_Grid_Bucket(const int grid_numbers, int *atom_numbers_in_grid_bucket,
                                         int *bucket, int max_atom_in_grid_numbers) {
  int grid_serial = blockDim.x * blockIdx.x + threadIdx.x;
  if (grid_serial < grid_numbers) {
    for (int i = 0; i < atom_numbers_in_grid_bucket[grid_serial]; i = i + 1) {
      bucket[grid_serial * max_atom_in_grid_numbers + i] = -1;
    }
    atom_numbers_in_grid_bucket[grid_serial] = 0;
  }
}

static __global__ void Put_Atom_Into_Bucket(const int atom_numbers, const int max_atom_in_grid_numbers,
                                            const int *atom_in_grid_serial,
                                            int *bucket, int *atom_numbers_in_grid_bucket) {
  int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (atom_i < atom_numbers) {
    int grid_serial = atom_in_grid_serial[atom_i];
    int a = atom_numbers_in_grid_bucket[grid_serial];
    atomicCAS(&bucket[grid_serial * max_atom_in_grid_numbers + a], -1, atom_i);
    if (bucket[grid_serial * max_atom_in_grid_numbers + a] != atom_i) {
      while (true) {
        a = a + 1;
        atomicCAS(&bucket[grid_serial * max_atom_in_grid_numbers + a], -1, atom_i);
        if (bucket[grid_serial * max_atom_in_grid_numbers + a] == atom_i) {
          atomicAdd(&atom_numbers_in_grid_bucket[grid_serial], 1);
          break;
        }
      }
    } else {
      atomicAdd(&atom_numbers_in_grid_bucket[grid_serial], 1);
    }
  }
}

static __global__ void Copy_Bucket_To_Output(const int grid_numbers, int max_atom_in_grid_numbers, int *bucket,
                                             int *atom_numbers_in_grid_bucket, int *output_bucket,
                                             int *output_atom_numbers_in_grid_bucket) {
  int grid_serial = blockDim.x * blockIdx.x + threadIdx.x;
  if (grid_serial < grid_numbers) {
    for (int i = 0; i < max_atom_in_grid_numbers; i = i + 1) {
      output_bucket[grid_serial * max_atom_in_grid_numbers + i] = bucket[grid_serial * max_atom_in_grid_numbers + i];
    }
    output_atom_numbers_in_grid_bucket[grid_serial] = atom_numbers_in_grid_bucket[grid_serial];
  }
}

const int max_atom_in_grid_numbers = 64;
extern "C" int PutAtomIntoBucket(int nparam, void **params, int *ndims, int64_t **shapes,
                                 const char **dtypes, void *stream, void *extra) {
  int atom_number, grid_numbers;

  int *atom_in_grid_serial = static_cast<int *>(params[0]);
  int *bucket = static_cast<int *>(params[1]);
  int *atom_numbers_in_grid_bucket = static_cast<int *>(params[2]);
  int *output_bucket = static_cast<int *>(params[3]);
  int *output_atom_numbers_in_grid_bucket =  static_cast<int *>(params[4]);

  atom_number = shapes[0][0];
  grid_numbers = shapes[1][0];
  Clear_Grid_Bucket<<<ceilf(static_cast<float>(grid_numbers) / 32), 32, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
    grid_numbers, atom_numbers_in_grid_bucket, bucket, max_atom_in_grid_numbers);
  Put_Atom_Into_Bucket<<<ceilf(static_cast<float>(atom_number) / 32), 32, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
    atom_number, max_atom_in_grid_numbers, atom_in_grid_serial, bucket, atom_numbers_in_grid_bucket);
  Copy_Bucket_To_Output<<<ceilf(static_cast<float>(grid_numbers) / 32), 32, 0,
    reinterpret_cast<cudaStream_t>(stream)>>>(grid_numbers, max_atom_in_grid_numbers, bucket,
    atom_numbers_in_grid_bucket, output_bucket, output_atom_numbers_in_grid_bucket);
  return 0;
}
