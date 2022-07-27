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

#include "common.cuh"

__device__ __host__ VECTOR operator+(const VECTOR &veca, const VECTOR &vecb) {
  VECTOR vec;
  vec.x = veca.x + vecb.x;
  vec.y = veca.y + vecb.y;
  vec.z = veca.z + vecb.z;
  return vec;
}

__device__ __host__ VECTOR operator+(const VECTOR &veca, const float &b) {
  VECTOR vec;
  vec.x = veca.x + b;
  vec.y = veca.y + b;
  vec.z = veca.z + b;
  return vec;
}

__device__ __host__ float operator*(const VECTOR &veca, const VECTOR &vecb) {
  return veca.x * vecb.x + veca.y * vecb.y + veca.z * vecb.z;
}
__device__ __host__ VECTOR operator*(const float &a, const VECTOR &vecb) {
  VECTOR vec;
  vec.x = a * vecb.x;
  vec.y = a * vecb.y;
  vec.z = a * vecb.z;
  return vec;
}

__device__ __host__ VECTOR operator-(const VECTOR &veca, const VECTOR &vecb) {
  VECTOR vec;
  vec.x = veca.x - vecb.x;
  vec.y = veca.y - vecb.y;
  vec.z = veca.z - vecb.z;
  return vec;
}

__device__ __host__ VECTOR operator-(const VECTOR &veca, const float &b) {
  VECTOR vec;
  vec.x = veca.x - b;
  vec.y = veca.y - b;
  vec.z = veca.z - b;
  return vec;
}

__device__ __host__ VECTOR operator-(const VECTOR &vecb) {
  VECTOR vec;
  vec.x = -vecb.x;
  vec.y = -vecb.y;
  vec.z = -vecb.z;
  return vec;
}

__device__ __host__ VECTOR operator/(const VECTOR &veca, const VECTOR &vecb) {
  VECTOR vec;
  vec.x = veca.x / vecb.x;
  vec.y = veca.y / vecb.y;
  vec.z = veca.z / vecb.z;
  return vec;
}

__device__ __host__ VECTOR operator/(const float &a, const VECTOR &vecb) {
  VECTOR vec;
  vec.x = a / vecb.x;
  vec.y = a / vecb.y;
  vec.z = a / vecb.z;
  return vec;
}

__device__ __host__ VECTOR operator^(const VECTOR &veca, const VECTOR &vecb) {
  VECTOR vec;
  vec.x = veca.y * vecb.z - veca.z * vecb.y;
  vec.y = veca.z * vecb.x - veca.x * vecb.z;
  vec.z = veca.x * vecb.y - veca.y * vecb.x;
  return vec;
}

__device__ __host__ VECTOR Get_Periodic_Displacement(
    const UNSIGNED_INT_VECTOR uvec_a, const UNSIGNED_INT_VECTOR uvec_b,
    const VECTOR scaler) {
  VECTOR dr;
  dr.x = ((int)(uvec_a.uint_x - uvec_b.uint_x)) * scaler.x;
  dr.y = ((int)(uvec_a.uint_y - uvec_b.uint_y)) * scaler.y;
  dr.z = ((int)(uvec_a.uint_z - uvec_b.uint_z)) * scaler.z;
  return dr;
}

__device__ __host__ VECTOR Get_Periodic_Displacement(const VECTOR vec_a,
                                                     const VECTOR vec_b,
                                                     const VECTOR box_length) {
  VECTOR dr;
  dr = vec_a - vec_b;
  dr.x = dr.x - floorf(dr.x / box_length.x + 0.5) * box_length.x;
  dr.y = dr.y - floorf(dr.y / box_length.y + 0.5) * box_length.y;
  dr.z = dr.z - floorf(dr.z / box_length.z + 0.5) * box_length.z;
  return dr;
}

__device__ __host__ VECTOR Get_Periodic_Displacement(
    const VECTOR vec_a, const VECTOR vec_b, const VECTOR box_length,
    const VECTOR box_length_inverse) {
  VECTOR dr;
  dr = vec_a - vec_b;
  dr.x = dr.x - floorf(dr.x * box_length_inverse.x + 0.5) * box_length.x;
  dr.y = dr.y - floorf(dr.y * box_length_inverse.y + 0.5) * box_length.y;
  dr.z = dr.z - floorf(dr.z * box_length_inverse.z + 0.5) * box_length.z;
  return dr;
}

__device__ VECTOR Make_Vector_Not_Exceed_Value(VECTOR vector,
                                               const float value) {
  return fminf(1.0, value * rnorm3df(vector.x, vector.y, vector.z)) * vector;
}

void Reset_List(int *list, const int replace_element, const int element_numbers,
                const int threads) {
  Reset_List<<<(unsigned int)ceilf((float)element_numbers / threads),
               threads>>>(element_numbers, list, replace_element);
}

void Reset_List(float *list, const float replace_element,
                const int element_numbers, const int threads) {
  Reset_List<<<(unsigned int)ceilf((float)element_numbers / threads),
               threads>>>(element_numbers, list, replace_element);
}

__global__ void Reset_List(const int element_numbers, int *list,
                           const int replace_element) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < element_numbers) {
    list[i] = replace_element;
  }
}
__global__ void Reset_List(const int element_numbers, float *list,
                           const float replace_element) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < element_numbers) {
    list[i] = replace_element;
  }
}

void Scale_List(float *list, const float scaler, const int element_numbers,
                int threads) {
  Scale_List<<<(unsigned int)ceilf((float)element_numbers / threads),
               threads>>>(element_numbers, list, scaler);
}

__global__ void Scale_List(const int element_numbers, float *list,
                           float scaler) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < element_numbers) {
    list[i] = list[i] * scaler;
  }
}
__global__ void Copy_List(const int element_numbers, const int *origin_list,
                          int *list) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < element_numbers) {
    list[i] = origin_list[i];
  }
}
__global__ void Copy_List(const int element_numbers, const float *origin_list,
                          float *list) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < element_numbers) {
    list[i] = origin_list[i];
  }
}
__global__ void Inverse_List_Element(const int element_numbers,
                                     const float *origin_list, float *list) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < element_numbers) {
    list[i] = 1. / origin_list[i];
  }
}

void Sum_Of_List(const int *list, int *sum, const int element_numbers,
                 int threads) {
  Sum_Of_List<<<1, threads>>>(element_numbers, list, sum);
}

void Sum_Of_List(const float *list, float *sum, const int end, const int start,
                 int threads) {
  Sum_Of_List<<<1, threads>>>(start, end, list, sum);
}

__global__ void Sum_Of_List(const int element_numbers, const int *list,
                            int *sum) {
  if (threadIdx.x == 0) {
    sum[0] = 0;
  }
  __syncthreads();
  int lin = 0;
  for (int i = threadIdx.x; i < element_numbers; i = i + blockDim.x) {
    lin = lin + list[i];
  }
  atomicAdd(sum, lin);
}
__global__ void Sum_Of_List(const int start, const int end, const float *list,
                            float *sum) {
  if (threadIdx.x == 0) {
    sum[0] = 0.;
  }
  __syncthreads();
  float lin = 0.;
  for (int i = threadIdx.x + start; i < end; i = i + blockDim.x) {
    lin = lin + list[i];
  }
  atomicAdd(sum, lin);
}
__global__ void Sum_Of_List(const int element_numbers, const float *list,
                            float *sum) {
  if (threadIdx.x == 0) {
    sum[0] = 0.;
  }
  __syncthreads();
  float lin = 0.;
  for (int i = threadIdx.x; i < element_numbers; i = i + blockDim.x) {
    lin = lin + list[i];
  }
  atomicAdd(sum, lin);
}
__global__ void Sum_Of_List(const int element_numbers, const VECTOR *list,
                            VECTOR *sum) {
  if (threadIdx.x == 0) {
    sum[0].x = 0.;
    sum[0].y = 0.;
    sum[0].z = 0.;
  }
  __syncthreads();
  VECTOR lin = {0., 0., 0.};
  for (int i = threadIdx.x; i < element_numbers; i = i + blockDim.x) {
    lin.x = lin.x + list[i].x;
    lin.y = lin.y + list[i].y;
    lin.z = lin.z + list[i].z;
  }
  atomicAdd(&sum[0].x, lin.x);
  atomicAdd(&sum[0].y, lin.y);
  atomicAdd(&sum[0].z, lin.z);
}
__global__ void Crd_To_Uint_Crd(const int atom_numbers,
                                const VECTOR scale_factor, const VECTOR *crd,
                                UNSIGNED_INT_VECTOR *uint_crd) {
  int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (atom_i < atom_numbers) {
    INT_VECTOR tempi;
    VECTOR temp = crd[atom_i];

    temp.x *= scale_factor.x;
    temp.y *= scale_factor.y;
    temp.z *= scale_factor.z;

    tempi.int_x = temp.x;
    tempi.int_y = temp.y;
    tempi.int_z = temp.z;

    uint_crd[atom_i].uint_x = (tempi.int_x << 2);
    uint_crd[atom_i].uint_y = (tempi.int_y << 2);
    uint_crd[atom_i].uint_z = (tempi.int_z << 2);
  }
}
__global__ void Crd_To_Int_Crd(const int atom_numbers,
                               const VECTOR scale_factor, const VECTOR *crd,
                               INT_VECTOR *int_crd) {
  int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (atom_i < atom_numbers) {
    int_crd[atom_i].int_x = (float)crd[atom_i].x * scale_factor.x;
    int_crd[atom_i].int_y = (float)crd[atom_i].y * scale_factor.y;
    int_crd[atom_i].int_z = (float)crd[atom_i].z * scale_factor.z;
  }
}
__global__ void Crd_Periodic_Map(const int atom_numbers, VECTOR *crd,
                                 const VECTOR box_length) {
  int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (atom_i < atom_numbers) {
    if (crd[atom_i].x >= 0) {
      if (crd[atom_i].x < box_length.x) {
      } else {
        crd[atom_i].x = crd[atom_i].x - box_length.x;
      }
    } else {
      crd[atom_i].x = crd[atom_i].x + box_length.x;
    }

    if (crd[atom_i].y >= 0) {
      if (crd[atom_i].y < box_length.y) {
      } else {
        crd[atom_i].y = crd[atom_i].y - box_length.y;
      }
    } else {
      crd[atom_i].y = crd[atom_i].y + box_length.y;
    }

    if (crd[atom_i].z >= 0) {
      if (crd[atom_i].z < box_length.z) {
      } else {
        crd[atom_i].z = crd[atom_i].z - box_length.z;
      }
    } else {
      crd[atom_i].z = crd[atom_i].z + box_length.z;
    }
  }
}

void Vector_Translation(const int vector_numbers, VECTOR *vec_list,
                        const VECTOR translation_vec, int threads_per_block) {
  Vector_Translation<<<(unsigned int)ceilf((float)vector_numbers /
                                           threads_per_block),
                       threads_per_block>>>(vector_numbers, vec_list,
                                            translation_vec);
}

void Vector_Translation(const int vector_numbers, VECTOR *vec_list,
                        const VECTOR *translation_vec, int threads_per_block) {
  Vector_Translation<<<(unsigned int)ceilf((float)vector_numbers /
                                           threads_per_block),
                       threads_per_block>>>(vector_numbers, vec_list,
                                            translation_vec);
}

__global__ void Vector_Translation(const int vector_numbers, VECTOR *vec_list,
                                   const VECTOR translation_vec) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < vector_numbers) {
    vec_list[i].x = vec_list[i].x + translation_vec.x;
    vec_list[i].y = vec_list[i].y + translation_vec.y;
    vec_list[i].z = vec_list[i].z + translation_vec.z;
  }
}
__global__ void Vector_Translation(const int vector_numbers, VECTOR *vec_list,
                                   const VECTOR *translation_vec) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < vector_numbers) {
    vec_list[i].x = vec_list[i].x + translation_vec[0].x;
    vec_list[i].y = vec_list[i].y + translation_vec[0].y;
    vec_list[i].z = vec_list[i].z + translation_vec[0].z;
  }
}

bool Malloc_Safely(void **address, size_t size) {
  address[0] = NULL;
  address[0] = (void *)malloc(size);
  if (address[0] != NULL) {
    return true;
  } else {
    printf("malloc failed!\n");
    getchar();
    return false;
  }
}
bool Cuda_Malloc_Safely(void **address, size_t size) {
  cudaError_t cuda_error = cudaMalloc(&address[0], size);
  if (cuda_error == 0) {
    return true;
  } else {
    printf("cudaMalloc failed! error %d\n", cuda_error);
    getchar();
    return false;
  }
}
bool Open_File_Safely(FILE **file, const char *file_name,
                      const char *open_type) {
  file[0] = NULL;
  file[0] = fopen(file_name, open_type);
  if (file[0] == NULL) {
    printf("Open file %s failed.\n", file_name);
    getchar();
    return false;
  } else {
    return true;
  }
}

__global__ void Setup_Rand_Normal_Kernel(const int float4_numbers,
                                         curandStatePhilox4_32_10_t *rand_state,
                                         const int seed) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  /* Each thread gets same seed, a different sequence
  number, no offset */
  if (id < float4_numbers) {
    curand_init(seed, id, 0, &rand_state[id]);
  }
}
__global__ void Rand_Normal(const int float4_numbers,
                            curandStatePhilox4_32_10_t *rand_state,
                            float4 *rand_float4) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < float4_numbers) {
    rand_float4[i] = curand_normal4(&rand_state[i]);
  }
}

__global__ void Cuda_Debug_Print(float *x) { printf("DEBUG: %f\n", x[0]); }

__global__ void Cuda_Debug_Print(VECTOR *x) {
  printf("DEBUG: %f %f %f\n", x[0].x, x[0].y, x[0].z);
}

__global__ void Cuda_Debug_Print(int *x) { printf("DEBUG: %d\n", x[0]); }

int Check_2357_Factor(int number) {
  int tempn;
  while (number > 0) {
    if (number == 1)
      return 1;
    tempn = number / 2;
    if (tempn * 2 != number)
      break;
    number = tempn;
  }

  while (number > 0) {
    if (number == 1)
      return 1;
    tempn = number / 3;
    if (tempn * 3 != number)
      break;
    number = tempn;
  }

  while (number > 0) {
    if (number == 1)
      return 1;
    tempn = number / 5;
    if (tempn * 5 != number)
      break;
    number = tempn;
  }

  while (number > 0) {
    if (number == 1)
      return 1;
    tempn = number / 7;
    if (tempn * 7 != number)
      break;
    number = tempn;
  }

  return 0;
}

int Get_Fft_Patameter(float length) {
  int tempi = (int)ceil(length + 3) >> 2 << 2;

  if (tempi >= 60 && tempi <= 68)
    tempi = 64;
  else if (tempi >= 120 && tempi <= 136)
    tempi = 128;
  else if (tempi >= 240 && tempi <= 272)
    tempi = 256;
  else if (tempi >= 480 && tempi <= 544)
    tempi = 512;
  else if (tempi >= 960 && tempi <= 1088)
    tempi = 1024;

  while (1) {
    if (Check_2357_Factor(tempi))
      return tempi;
    tempi += 4;
  }
}
