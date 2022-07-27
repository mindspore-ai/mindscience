/*
 * Copyright 2021 Gao's lab, Peking University, CCME. All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "crd_molecular_map.cuh"

__global__ void Calculate_No_Wrap_Crd_CUDA(const int atom_numbers,
                                           const INT_VECTOR *box_map_times,
                                           const VECTOR box, const VECTOR *crd,
                                           VECTOR *nowrap_crd) {
  for (int i = threadIdx.x; i < atom_numbers; i = i + blockDim.x) {
    nowrap_crd[i].x = (float)box_map_times[i].int_x * box.x + crd[i].x;
    nowrap_crd[i].y = (float)box_map_times[i].int_y * box.y + crd[i].y;
    nowrap_crd[i].z = (float)box_map_times[i].int_z * box.z + crd[i].z;
  }
}

__global__ void Refresh_BoxMapTimes_CUDA(const int atom_numbers,
                                         const VECTOR box_length_inverse,
                                         const VECTOR *crd,
                                         INT_VECTOR *box_map_times,
                                         VECTOR *old_crd) {
  VECTOR crd_i, old_crd_i;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < atom_numbers;
       i += gridDim.x * blockDim.x) {
    crd_i = crd[i];
    old_crd_i = old_crd[i];
    box_map_times[i].int_x +=
        floor((old_crd_i.x - crd_i.x) * box_length_inverse.x + 0.5);
    box_map_times[i].int_y +=
        floor((old_crd_i.y - crd_i.y) * box_length_inverse.y + 0.5);
    box_map_times[i].int_z +=
        floor((old_crd_i.z - crd_i.z) * box_length_inverse.z + 0.5);
    old_crd[i] = crd_i;
  }
}

void Move_Crd_Nearest_From_Exclusions_Host(int atom_numbers, VECTOR *crd,
                                           INT_VECTOR *box_map_times,
                                           const VECTOR box_length,
                                           const int exclude_numbers,
                                           const int *exclude_length,
                                           const int *exclude_start,
                                           const int *exclude_list) {
  //分子拓扑是一个无向图，邻接表进行描述，通过排除表形成
  int edge_numbers = 2 * exclude_numbers;
  int *visited = NULL;    //每个原子是否拜访过
  int *first_edge = NULL; //每个原子的第一个边（链表的头）
  int *edges = NULL;      //每个边的序号
  int *edge_next = NULL;  //每个原子的边（链表结构）
  Malloc_Safely((void **)&visited, sizeof(int) * atom_numbers);
  Malloc_Safely((void **)&first_edge, sizeof(int) * atom_numbers);
  Malloc_Safely((void **)&edges, sizeof(int) * edge_numbers);
  Malloc_Safely((void **)&edge_next, sizeof(int) * edge_numbers);
  //初始化链表
  for (int i = 0; i < atom_numbers; i++) {
    visited[i] = 0;
    first_edge[i] = -1;
  }
  int atom_i, atom_j, edge_count = 0;
  for (int i = 0; i < atom_numbers; i++) {
    atom_i = i;
    for (int j = exclude_start[i] + exclude_length[i] - 1;
         j >= exclude_start[i]; j--) {
      //这里使用倒序是因为链表构建是用的头插法
      atom_j = exclude_list[j];
      edge_next[edge_count] = first_edge[atom_i];
      first_edge[atom_i] = edge_count;
      edges[edge_count] = atom_j;
      edge_count++;
      edge_next[edge_count] = first_edge[atom_j];
      first_edge[atom_j] = edge_count;
      edges[edge_count] = atom_i;
      edge_count++;
    }
  }
  std::deque<int> queue;
  int atom, atom_front;
  for (int i = 0; i < atom_numbers; i++) {
    if (!visited[i]) {
      visited[i] = 1;
      queue.push_back(i);
      atom_front = i;
      while (!queue.empty()) {
        atom = queue[0];
        queue.pop_front();
        box_map_times[atom].int_x =
            floorf((crd[atom_front].x - crd[atom].x) / box_length.x + 0.5);
        box_map_times[atom].int_y =
            floorf((crd[atom_front].y - crd[atom].y) / box_length.y + 0.5);
        box_map_times[atom].int_z =
            floorf((crd[atom_front].z - crd[atom].z) / box_length.z + 0.5);
        crd[atom].x = crd[atom].x + box_map_times[atom].int_x * box_length.x;
        crd[atom].y = crd[atom].y + box_map_times[atom].int_y * box_length.y;
        crd[atom].z = crd[atom].z + box_map_times[atom].int_z * box_length.z;
        edge_count = first_edge[atom];
        atom_front = atom;

        while (edge_count != -1) {
          atom = edges[edge_count];
          if (!visited[atom]) {
            queue.push_back(atom);
            visited[atom] = 1;
          }
          edge_count = edge_next[edge_count];
        }
      }
    }
  }
  free(visited);
  free(first_edge);
  free(edges);
  free(edge_next);
}

void CoordinateMolecularMap::Record_Box_Map_Times_Host(
    int atom_numbers, VECTOR *crd, VECTOR *old_crd, INT_VECTOR *box_map_times,
    VECTOR box) {
  for (int i = 0; i < atom_numbers; i = i + 1) {
    box_map_times[i].int_x +=
        floor((old_crd[i].x - crd[i].x) / box_length.x + 0.5);
    box_map_times[i].int_y +=
        floor((old_crd[i].y - crd[i].y) / box_length.y + 0.5);
    box_map_times[i].int_z +=
        floor((old_crd[i].z - crd[i].z) / box_length.z + 0.5);
  }
}

void CoordinateMolecularMap::Initial(int atom_numbers, VECTOR box_length,
                                     VECTOR *crd, const int exclude_numbers,
                                     const int *exclude_length,
                                     const int *exclude_start,
                                     const int *exclude_list,
                                     const char *module_name) {
  if (module_name == NULL) {
    strcpy(this->module_name, "crd_mole_wrap");
  } else {
    strcpy(this->module_name, module_name);
  }

  this->atom_numbers = atom_numbers;
  this->box_length = box_length;

  VECTOR *coordinate = NULL;
  Malloc_Safely((void **)&coordinate, sizeof(VECTOR) * atom_numbers);
  cudaMemcpy(coordinate, crd, sizeof(VECTOR) * atom_numbers,
             cudaMemcpyDeviceToHost);

  Cuda_Malloc_Safely((void **)&nowrap_crd, sizeof(VECTOR) * atom_numbers);
  Cuda_Malloc_Safely((void **)&old_crd, sizeof(VECTOR) * atom_numbers);
  Cuda_Malloc_Safely((void **)&box_map_times,
                     sizeof(INT_VECTOR) * atom_numbers);

  Malloc_Safely((void **)&h_nowrap_crd, sizeof(VECTOR) * atom_numbers);
  Malloc_Safely((void **)&h_old_crd, sizeof(VECTOR) * atom_numbers);
  Malloc_Safely((void **)&h_box_map_times, sizeof(INT_VECTOR) * atom_numbers);

  for (int i = 0; i < atom_numbers; i = i + 1) {
    h_old_crd[i] = coordinate[i];
    h_nowrap_crd[i] = coordinate[i];
    h_box_map_times[i].int_x = 0;
    h_box_map_times[i].int_y = 0;
    h_box_map_times[i].int_z = 0;
  }

  Move_Crd_Nearest_From_Exclusions_Host(
      atom_numbers, h_nowrap_crd, h_box_map_times, box_length, exclude_numbers,
      exclude_length, exclude_start, exclude_list);

  //使用cuda内部函数，给出占用率最大的block和thread参数
  cudaOccupancyMaxPotentialBlockSize(&blocks_per_grid, &threads_per_block,
                                     Refresh_BoxMapTimes_CUDA, 0, 0);

  cudaMemcpy(nowrap_crd, h_nowrap_crd, sizeof(VECTOR) * atom_numbers,
             cudaMemcpyHostToDevice);
  cudaMemcpy(old_crd, h_old_crd, sizeof(VECTOR) * atom_numbers,
             cudaMemcpyHostToDevice);
  cudaMemcpy(box_map_times, h_box_map_times, sizeof(INT_VECTOR) * atom_numbers,
             cudaMemcpyHostToDevice);
  free(coordinate);
  is_initialized = 1;
}

void CoordinateMolecularMap::Calculate_No_Wrap_Crd(const VECTOR *crd) {
  if (is_initialized)
    Calculate_No_Wrap_Crd_CUDA<<<blocks_per_grid, threads_per_block>>>(
        atom_numbers, box_map_times, box_length, crd, nowrap_crd);
}

void CoordinateMolecularMap::Refresh_BoxMapTimes(const VECTOR *crd) {
  if (is_initialized) {
    Refresh_BoxMapTimes_CUDA<<<blocks_per_grid, threads_per_block>>>(
        atom_numbers, 1.0 / box_length, crd, box_map_times, old_crd);
  }
}

void CoordinateMolecularMap::Update_Volume(VECTOR box_length) {
  if (!is_initialized)
    return;
  this->box_length = box_length;
}
