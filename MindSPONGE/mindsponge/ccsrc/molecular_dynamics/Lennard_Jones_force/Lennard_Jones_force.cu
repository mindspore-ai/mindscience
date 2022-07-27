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

#include "Lennard_Jones_force.cuh"

#define TWO_DIVIDED_BY_SQRT_PI 1.1283791670218446

//由LJ坐标和转化系数求距离
__device__ __host__ VECTOR Get_Periodic_Displacement(
    const UINT_VECTOR_LJ_TYPE uvec_a, const UINT_VECTOR_LJ_TYPE uvec_b,
    const VECTOR scaler) {
  VECTOR dr;
  dr.x = ((int)(uvec_a.uint_x - uvec_b.uint_x)) * scaler.x;
  dr.y = ((int)(uvec_a.uint_y - uvec_b.uint_y)) * scaler.y;
  dr.z = ((int)(uvec_a.uint_z - uvec_b.uint_z)) * scaler.z;
  return dr;
}

__global__ void Copy_LJ_Type_To_New_Crd(const int atom_numbers,
                                        UINT_VECTOR_LJ_TYPE *new_crd,
                                        const int *LJ_type) {
  int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (atom_i < atom_numbers) {
    new_crd[atom_i].LJ_type = LJ_type[atom_i];
  }
}

__global__ void Copy_Crd_And_Charge_To_New_Crd(const int atom_numbers,
                                               const UNSIGNED_INT_VECTOR *crd,
                                               UINT_VECTOR_LJ_TYPE *new_crd,
                                               const float *charge) {
  int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (atom_i < atom_numbers) {
    new_crd[atom_i].uint_x = crd[atom_i].uint_x;
    new_crd[atom_i].uint_y = crd[atom_i].uint_y;
    new_crd[atom_i].uint_z = crd[atom_i].uint_z;
    new_crd[atom_i].charge = charge[atom_i];
  }
}

__global__ void Copy_Crd_To_New_Crd(const int atom_numbers,
                                    const UNSIGNED_INT_VECTOR *crd,
                                    UINT_VECTOR_LJ_TYPE *new_crd) {
  int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (atom_i < atom_numbers) {
    new_crd[atom_i].uint_x = crd[atom_i].uint_x;
    new_crd[atom_i].uint_y = crd[atom_i].uint_y;
    new_crd[atom_i].uint_z = crd[atom_i].uint_z;
  }
}

static __global__ void LJ_Force_With_Direct_CF_CUDA(
    const int atom_numbers, const ATOM_GROUP *nl,
    const UINT_VECTOR_LJ_TYPE *uint_crd, const VECTOR boxlength,
    const float *LJ_type_A, const float *LJ_type_B, const float cutoff,
    VECTOR *frc, const float pme_beta, const float sqrt_pi) {
  int atom_i = blockDim.y * blockIdx.y + threadIdx.y;
  if (atom_i < atom_numbers) {
    ATOM_GROUP nl_i = nl[atom_i];
    int N = nl_i.atom_numbers;
    int atom_j;
    int int_x;
    int int_y;
    int int_z;
    UINT_VECTOR_LJ_TYPE r1 = uint_crd[atom_i], r2;
    VECTOR dr;
    float dr_2;
    float dr_4;
    float dr_8;
    float dr_6;
    float frc_abs = 0.;
    VECTOR frc_lin;
    VECTOR frc_record = {0., 0., 0.};

    // CF
    float charge_i = r1.charge; // r1.charge;
    float charge_j;
    float dr_abs;
    float dr_1;
    float beta_dr;
    float frc_cf_abs;
    //

    int x, y;
    int atom_pair_LJ_type;
    for (int j = threadIdx.x; j < N; j = j + blockDim.x) {
      atom_j = nl_i.atom_serial[j];
      r2 = uint_crd[atom_j];
      // CF
      charge_j = r2.charge;

      int_x = r2.uint_x - r1.uint_x;
      int_y = r2.uint_y - r1.uint_y;
      int_z = r2.uint_z - r1.uint_z;
      dr.x = boxlength.x * int_x;
      dr.y = boxlength.y * int_y;
      dr.z = boxlength.z * int_z;
      dr_abs = norm3df(dr.x, dr.y, dr.z);
      if (dr_abs < cutoff) {
        dr_1 = 1. / dr_abs;
        dr_2 = dr_1 * dr_1;
        dr_4 = dr_2 * dr_2;
        dr_8 = dr_4 * dr_4;
        dr_6 = dr_4 * dr_2;

        y = (r2.LJ_type - r1.LJ_type);
        x = y >> 31;
        y = (y ^ x) - x;
        x = r2.LJ_type + r1.LJ_type;
        r2.LJ_type = (x + y) >> 1;
        x = (x - y) >> 1;
        atom_pair_LJ_type = (r2.LJ_type * (r2.LJ_type + 1) >> 1) + x;

        frc_abs = (-LJ_type_A[atom_pair_LJ_type] * dr_6 +
                   LJ_type_B[atom_pair_LJ_type]) *
                  dr_8;
        // PME的直接部分
        beta_dr = pme_beta * dr_abs;
        // sqrt_pi= 2/sqrt(3.141592654);
        frc_cf_abs =
            beta_dr * sqrt_pi * expf(-beta_dr * beta_dr) + erfcf(beta_dr);
        frc_cf_abs = frc_cf_abs * dr_2 * dr_1;
        frc_cf_abs = charge_i * charge_j * frc_cf_abs;

        frc_abs = frc_abs - frc_cf_abs;

        frc_lin.x = frc_abs * dr.x;
        frc_lin.y = frc_abs * dr.y;
        frc_lin.z = frc_abs * dr.z;

        frc_record.x = frc_record.x + frc_lin.x;
        frc_record.y = frc_record.y + frc_lin.y;
        frc_record.z = frc_record.z + frc_lin.z;

        atomicAdd(&frc[atom_j].x, -frc_lin.x);
        atomicAdd(&frc[atom_j].y, -frc_lin.y);
        atomicAdd(&frc[atom_j].z, -frc_lin.z);
      }
    } // atom_j cycle

    int delta = 32;
    for (int i = 0; i < 5; i += 1) {
      delta >>= 1;
      frc_record.x += __shfl_down_sync(0xFFFFFFFF, frc_record.x, delta, 32);
      frc_record.y += __shfl_down_sync(0xFFFFFFFF, frc_record.y, delta, 32);
      frc_record.z += __shfl_down_sync(0xFFFFFFFF, frc_record.z, delta, 32);
    }
    if (threadIdx.x == 0) {
      atomicAdd(&frc[atom_i].x, frc_record.x);
      atomicAdd(&frc[atom_i].y, frc_record.y);
      atomicAdd(&frc[atom_i].z, frc_record.z);
    }
  }
}

static __global__ void device_add(float *variable, const float adder) {
  variable[0] += adder;
}

static __global__ void LJ_Direct_CF_Force_With_Atom_Energy_CUDA(
    const int atom_numbers, const ATOM_GROUP *nl,
    const UINT_VECTOR_LJ_TYPE *uint_crd, const VECTOR boxlength,
    const float *LJ_type_A, const float *LJ_type_B, const float cutoff,
    VECTOR *frc, const float pme_beta, const float sqrt_pi,
    float *atom_energy) {
  int atom_i = blockDim.y * blockIdx.y + threadIdx.y;
  if (atom_i < atom_numbers) {
    ATOM_GROUP nl_i = nl[atom_i];
    int N = nl_i.atom_numbers;
    int atom_j;
    int int_x;
    int int_y;
    int int_z;
    UINT_VECTOR_LJ_TYPE r1 = uint_crd[atom_i], r2;
    VECTOR dr;
    float dr_2;
    float dr_4;
    float dr_8;
    float dr_6;
    float frc_abs = 0.;
    VECTOR frc_lin;
    VECTOR frc_record = {0., 0., 0.};

    // CF
    float charge_i = r1.charge; // r1.charge;
    float charge_j;
    float dr_abs;
    float dr_1;
    float beta_dr;
    float frc_cf_abs;

    //能量
    float ene_lin = 0.;
    float ene_lin2 = 0.;

    int x, y;
    int atom_pair_LJ_type;
    for (int j = threadIdx.x; j < N; j = j + blockDim.x) {
      atom_j = nl_i.atom_serial[j];
      r2 = uint_crd[atom_j];
      // CF
      charge_j = r2.charge;

      int_x = r2.uint_x - r1.uint_x;
      int_y = r2.uint_y - r1.uint_y;
      int_z = r2.uint_z - r1.uint_z;
      dr.x = boxlength.x * int_x;
      dr.y = boxlength.y * int_y;
      dr.z = boxlength.z * int_z;
      dr_abs = norm3df(dr.x, dr.y, dr.z);
      if (dr_abs < cutoff) {
        dr_1 = 1. / dr_abs;
        dr_2 = dr_1 * dr_1;
        dr_4 = dr_2 * dr_2;
        dr_8 = dr_4 * dr_4;
        dr_6 = dr_4 * dr_2;

        y = (r2.LJ_type - r1.LJ_type);
        x = y >> 31;
        y = (y ^ x) - x;
        x = r2.LJ_type + r1.LJ_type;
        r2.LJ_type = (x + y) >> 1;
        x = (x - y) >> 1;
        atom_pair_LJ_type = (r2.LJ_type * (r2.LJ_type + 1) >> 1) + x;

        frc_abs = (-LJ_type_A[atom_pair_LJ_type] * dr_6 +
                   LJ_type_B[atom_pair_LJ_type]) *
                  dr_8;
        // CF
        // dr_abs = sqrtf(dr2);
        beta_dr = pme_beta * dr_abs;
        // sqrt_pi= 2/sqrt(3.141592654);
        frc_cf_abs =
            beta_dr * sqrt_pi * expf(-beta_dr * beta_dr) + erfcf(beta_dr);
        frc_cf_abs = frc_cf_abs * dr_2 * dr_1;
        frc_cf_abs = charge_i * charge_j * frc_cf_abs;

        //能量
        ene_lin2 = ene_lin2 + charge_i * charge_j * erfcf(beta_dr) * dr_1;
        ene_lin = ene_lin + (0.083333333 * LJ_type_A[atom_pair_LJ_type] * dr_6 -
                             0.166666666 * LJ_type_B[atom_pair_LJ_type]) *
                                dr_6;

        //两种力的绝对值
        frc_abs = frc_abs - frc_cf_abs;

        frc_lin.x = frc_abs * dr.x;
        frc_lin.y = frc_abs * dr.y;
        frc_lin.z = frc_abs * dr.z;

        frc_record.x = frc_record.x + frc_lin.x;
        frc_record.y = frc_record.y + frc_lin.y;
        frc_record.z = frc_record.z + frc_lin.z;

        atomicAdd(&frc[atom_j].x, -frc_lin.x);
        atomicAdd(&frc[atom_j].y, -frc_lin.y);
        atomicAdd(&frc[atom_j].z, -frc_lin.z);
      }
    } // atom_j cycle
    ene_lin += ene_lin2;
    int delta = 32;
    for (int i = 0; i < 5; i += 1) {
      delta >>= 1;
      frc_record.x += __shfl_down_sync(0xFFFFFFFF, frc_record.x, delta, 32);
      frc_record.y += __shfl_down_sync(0xFFFFFFFF, frc_record.y, delta, 32);
      frc_record.z += __shfl_down_sync(0xFFFFFFFF, frc_record.z, delta, 32);
      ene_lin += __shfl_down_sync(0xFFFFFFFF, ene_lin, delta, 32);
    }
    if (threadIdx.x == 0) {
      atomicAdd(&frc[atom_i].x, frc_record.x);
      atomicAdd(&frc[atom_i].y, frc_record.y);
      atomicAdd(&frc[atom_i].z, frc_record.z);
      atom_energy[atom_i] += ene_lin;
    }
  }
}

static __global__ void LJ_Direct_CF_Force_With_LJ_Virial_Direct_CF_Energy_CUDA(
    const int atom_numbers, const ATOM_GROUP *nl,
    const UINT_VECTOR_LJ_TYPE *uint_crd, const VECTOR boxlength,
    const float *LJ_type_A, const float *LJ_type_B, const float cutoff,
    VECTOR *frc, const float pme_beta, const float sqrt_pi,
    float *atom_lj_virial, float *atom_direct_cf_energy) {
  int atom_i = blockDim.y * blockIdx.y + threadIdx.y;
  if (atom_i < atom_numbers) {
    ATOM_GROUP nl_i = nl[atom_i];
    int N = nl_i.atom_numbers;
    int atom_j;
    int int_x;
    int int_y;
    int int_z;
    UINT_VECTOR_LJ_TYPE r1 = uint_crd[atom_i], r2;
    VECTOR dr;
    float dr_2;
    float dr_4;
    float dr_8;
    float dr_6;
    float frc_abs = 0.;
    VECTOR frc_lin;
    VECTOR frc_record = {0., 0., 0.};

    // CF
    float charge_i = r1.charge; // r1.charge;
    float charge_j;
    float dr_abs;
    float dr_1;
    float beta_dr;
    float frc_cf_abs;
    //

    // LJ维里（未乘1/3/V）
    float virial_lin = 0.;
    // PME direct 能量
    float energy_lin = 0.;

    int x, y;
    int atom_pair_LJ_type;
    for (int j = threadIdx.x; j < N; j = j + blockDim.x) {
      atom_j = nl_i.atom_serial[j];
      r2 = uint_crd[atom_j];
      // CF
      charge_j = r2.charge;

      int_x = r2.uint_x - r1.uint_x;
      int_y = r2.uint_y - r1.uint_y;
      int_z = r2.uint_z - r1.uint_z;
      dr.x = boxlength.x * int_x;
      dr.y = boxlength.y * int_y;
      dr.z = boxlength.z * int_z;
      dr_abs = norm3df(dr.x, dr.y, dr.z);
      if (dr_abs < cutoff) {
        dr_1 = 1. / dr_abs;
        dr_2 = dr_1 * dr_1;
        dr_4 = dr_2 * dr_2;
        dr_8 = dr_4 * dr_4;
        dr_6 = dr_4 * dr_2;

        y = (r2.LJ_type - r1.LJ_type);
        x = y >> 31;
        y = (y ^ x) - x;
        x = r2.LJ_type + r1.LJ_type;
        r2.LJ_type = (x + y) >> 1;
        x = (x - y) >> 1;
        atom_pair_LJ_type = (r2.LJ_type * (r2.LJ_type + 1) >> 1) + x;

        frc_abs = (-LJ_type_A[atom_pair_LJ_type] * dr_6 +
                   LJ_type_B[atom_pair_LJ_type]) *
                  dr_8;
        // CF
        // dr_abs = sqrtf(dr2);
        beta_dr = pme_beta * dr_abs;
        // sqrt_pi= 2/sqrt(3.141592654);
        frc_cf_abs =
            beta_dr * sqrt_pi * expf(-beta_dr * beta_dr) + erfcf(beta_dr);
        frc_cf_abs = frc_cf_abs * dr_2 * dr_1;
        frc_cf_abs = charge_i * charge_j * frc_cf_abs;

        // PME的Direct静电能量
        energy_lin = energy_lin + charge_i * charge_j * erfcf(beta_dr) * dr_1;
        // LJ的dU/dr*r,frc_abs=dU/dr/r
        virial_lin = virial_lin - frc_abs * dr_abs * dr_abs;
        //两种力的绝对值
        frc_abs = frc_abs - frc_cf_abs;

        frc_lin.x = frc_abs * dr.x;
        frc_lin.y = frc_abs * dr.y;
        frc_lin.z = frc_abs * dr.z;

        frc_record.x = frc_record.x + frc_lin.x;
        frc_record.y = frc_record.y + frc_lin.y;
        frc_record.z = frc_record.z + frc_lin.z;

        atomicAdd(&frc[atom_j].x, -frc_lin.x);
        atomicAdd(&frc[atom_j].y, -frc_lin.y);
        atomicAdd(&frc[atom_j].z, -frc_lin.z);
      }
    } // atom_j cycle

    int delta = 32;
    for (int i = 0; i < 5; i += 1) {
      delta >>= 1;
      frc_record.x += __shfl_down_sync(0xFFFFFFFF, frc_record.x, delta, 32);
      frc_record.y += __shfl_down_sync(0xFFFFFFFF, frc_record.y, delta, 32);
      frc_record.z += __shfl_down_sync(0xFFFFFFFF, frc_record.z, delta, 32);
      energy_lin += __shfl_down_sync(0xFFFFFFFF, energy_lin, delta, 32);
      virial_lin += __shfl_down_sync(0xFFFFFFFF, virial_lin, delta, 32);
    }
    if (threadIdx.x == 0) {
      atomicAdd(&frc[atom_i].x, frc_record.x);
      atomicAdd(&frc[atom_i].y, frc_record.y);
      atomicAdd(&frc[atom_i].z, frc_record.z);
      atom_direct_cf_energy[atom_i] += energy_lin;
      atom_lj_virial[atom_i] += virial_lin;
    }
  }
}

static __global__ void
LJ_Direct_CF_Force_With_Atom_Energy_And_LJ_Virial_Direct_CF_Energy_CUDA(
    const int atom_numbers, const ATOM_GROUP *nl,
    const UINT_VECTOR_LJ_TYPE *uint_crd, const VECTOR boxlength,
    const float *LJ_type_A, const float *LJ_type_B, const float cutoff,
    VECTOR *frc, const float pme_beta, const float sqrt_pi, float *atom_energy,
    float *atom_lj_virial, float *atom_direct_cf_energy) {
  int atom_i = blockDim.y * blockIdx.y + threadIdx.y;
  if (atom_i < atom_numbers) {
    ATOM_GROUP nl_i = nl[atom_i];
    int N = nl_i.atom_numbers;
    int atom_j;
    int int_x;
    int int_y;
    int int_z;
    VECTOR frc_record = {0.0f, 0.0f, 0.0f};
    // LJ维里（未乘1/3/V）
    float virial_lin = 0.0f;
    // PME direct 能量
    float ene_lin2 = 0.0f;
    // lj能量
    float ene_lin = 0.0f;

    UINT_VECTOR_LJ_TYPE r1 = uint_crd[atom_i], r2;
    VECTOR dr;
    float dr_2;
    float dr_4;
    float dr_8;
    float dr_6;
    float frc_abs = 0.0f;
    int atom_pair_LJ_type;
    int x, y;
    VECTOR frc_lin;
    // CF
    float charge_i = r1.charge; // r1.charge;
    float charge_j;
    float dr_abs;
    float dr_1;
    float beta_dr;
    float frc_cf_abs;

    for (int j = threadIdx.x; j < N; j = j + blockDim.x) {
      atom_j = nl_i.atom_serial[j];
      r2 = uint_crd[atom_j];
      // CF
      charge_j = r2.charge;

      int_x = r2.uint_x - r1.uint_x;
      int_y = r2.uint_y - r1.uint_y;
      int_z = r2.uint_z - r1.uint_z;
      dr.x = boxlength.x * int_x;
      dr.y = boxlength.y * int_y;
      dr.z = boxlength.z * int_z;
      dr_abs = norm3df(dr.x, dr.y, dr.z);
      if (dr_abs < cutoff) {
        dr_1 = 1. / dr_abs;
        dr_2 = dr_1 * dr_1;
        dr_4 = dr_2 * dr_2;
        dr_8 = dr_4 * dr_4;
        dr_6 = dr_4 * dr_2;

        //这一步是通过位运算来通过LJ的atom_type获得正确的以下三角矩阵输入，一维数组储存的pair_type的值
        //实际上实现的效果等效于
        // r2.LJ_type与r1.LJ_type中，最大值赋给r2.LJ_type，最小值赋给x
        y = (r2.LJ_type - r1.LJ_type);
        x = y >> 31;
        y = (y ^ x) - x;
        x = r2.LJ_type + r1.LJ_type;
        r2.LJ_type = (x + y) >> 1;
        x = (x - y) >> 1;
        atom_pair_LJ_type = (r2.LJ_type * (r2.LJ_type + 1) >> 1) + x;

        frc_abs = (-LJ_type_A[atom_pair_LJ_type] * dr_6 +
                   LJ_type_B[atom_pair_LJ_type]) *
                  dr_8;

        // CF
        // dr_abs = sqrtf(dr2);
        beta_dr = pme_beta * dr_abs;
        // sqrt_pi= 2/sqrt(3.141592654);
        frc_cf_abs =
            beta_dr * sqrt_pi * expf(-beta_dr * beta_dr) + erfcf(beta_dr);
        frc_cf_abs = frc_cf_abs * dr_2 * dr_1;
        frc_cf_abs = charge_i * charge_j * frc_cf_abs;

        //能量
        ene_lin2 = ene_lin2 + charge_i * charge_j * erfcf(beta_dr) * dr_1;
        ene_lin = ene_lin + (0.083333333 * LJ_type_A[atom_pair_LJ_type] * dr_6 -
                             0.166666666 * LJ_type_B[atom_pair_LJ_type]) *
                                dr_6;

        // LJ的维里等于dU/dr*r，而frc_abs=dU/dr/r，因此转换
        virial_lin = virial_lin - frc_abs * dr_abs * dr_abs;

        //两种力的绝对值
        frc_abs = frc_abs - frc_cf_abs;

        frc_lin.x = frc_abs * dr.x;
        frc_lin.y = frc_abs * dr.y;
        frc_lin.z = frc_abs * dr.z;

        frc_record.x = frc_record.x + frc_lin.x;
        frc_record.y = frc_record.y + frc_lin.y;
        frc_record.z = frc_record.z + frc_lin.z;

        atomicAdd(&frc[atom_j].x, -frc_lin.x);
        atomicAdd(&frc[atom_j].y, -frc_lin.y);
        atomicAdd(&frc[atom_j].z, -frc_lin.z);
      }
    } // atom_j cycle
    ene_lin += ene_lin2;

    //通过原语对warp内的32个线程规约求和，减少atomicAdd的使用次数，以提高速度
    int delta = 32;
    for (int i = 0; i < 5; i += 1) {
      delta >>= 1;
      frc_record.x += __shfl_down_sync(0xFFFFFFFF, frc_record.x, delta, 32);
      frc_record.y += __shfl_down_sync(0xFFFFFFFF, frc_record.y, delta, 32);
      frc_record.z += __shfl_down_sync(0xFFFFFFFF, frc_record.z, delta, 32);
      ene_lin += __shfl_down_sync(0xFFFFFFFF, ene_lin, delta, 32);
      ene_lin2 += __shfl_down_sync(0xFFFFFFFF, ene_lin2, delta, 32);
      virial_lin += __shfl_down_sync(0xFFFFFFFF, virial_lin, delta, 32);
    }
    if (threadIdx.x == 0) {
      atomicAdd(&frc[atom_i].x, frc_record.x);
      atomicAdd(&frc[atom_i].y, frc_record.y);
      atomicAdd(&frc[atom_i].z, frc_record.z);
      atom_direct_cf_energy[atom_i] += ene_lin2;
      atom_energy[atom_i] += ene_lin;
      atom_lj_virial[atom_i] += virial_lin;
    }
  }
}

static __global__ void
LJ_Force_CUDA(const int atom_numbers, const ATOM_GROUP *nl,
              const UINT_VECTOR_LJ_TYPE *uint_crd, const VECTOR boxlength,
              const float *LJ_type_A, const float *LJ_type_B,
              const float cutoff, VECTOR *frc) {
  int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (atom_i < atom_numbers) {
    ATOM_GROUP nl_i = nl[atom_i];
    int N = nl_i.atom_numbers;
    // int B = (unsigned int)ceilf((float)N / blockDim.y);
    int atom_j;
    int int_x;
    int int_y;
    int int_z;
    UINT_VECTOR_LJ_TYPE r1 = uint_crd[atom_i], r2;
    VECTOR dr;
    // float dr2;
    float dr_2;
    float dr_4;
    float dr_8;
    float dr_6;
    float frc_abs = 0.0f;
    VECTOR frc_lin;
    VECTOR frc_record = {0.0f, 0.0f, 0.0f};
    float dr_abs;
    float dr_1;
    int x, y;
    int atom_pair_LJ_type;
    for (int j = threadIdx.y; j < N; j = j + blockDim.y) {
      atom_j = nl_i.atom_serial[j];
      r2 = uint_crd[atom_j];

      int_x = r2.uint_x - r1.uint_x;
      int_y = r2.uint_y - r1.uint_y;
      int_z = r2.uint_z - r1.uint_z;
      dr.x = boxlength.x * int_x;
      dr.y = boxlength.y * int_y;
      dr.z = boxlength.z * int_z;
      dr_abs = norm3df(dr.x, dr.y, dr.z);
      if (dr_abs < cutoff) {
        dr_1 = 1. / dr_abs;
        dr_2 = dr_1 * dr_1;
        dr_4 = dr_2 * dr_2;
        dr_8 = dr_4 * dr_4;
        // dr_14 = dr_8*dr_4*dr_2;
        dr_6 = dr_4 * dr_2;

        y = (r2.LJ_type - r1.LJ_type);
        x = y >> 31;
        y = (y ^ x) - x;
        x = r2.LJ_type + r1.LJ_type;
        r2.LJ_type = (x + y) >> 1;
        x = (x - y) >> 1;
        atom_pair_LJ_type = (r2.LJ_type * (r2.LJ_type + 1) >> 1) + x;

        // frc_abs = -LJ_type_A[atom_pair_LJ_type] * dr_14
        //           + LJ_type_B[atom_pair_LJ_type] * dr_8;

        frc_abs = (-LJ_type_A[atom_pair_LJ_type] * dr_6 +
                   LJ_type_B[atom_pair_LJ_type]) *
                  dr_8;

        //两种力的绝对值
        frc_abs = frc_abs;

        frc_lin.x = frc_abs * dr.x;
        frc_lin.y = frc_abs * dr.y;
        frc_lin.z = frc_abs * dr.z;

        frc_record.x = frc_record.x + frc_lin.x;
        frc_record.y = frc_record.y + frc_lin.y;
        frc_record.z = frc_record.z + frc_lin.z;

        atomicAdd(&frc[atom_j].x, -frc_lin.x);
        atomicAdd(&frc[atom_j].y, -frc_lin.y);
        atomicAdd(&frc[atom_j].z, -frc_lin.z);
      }
    } // atom_j cycle
    atomicAdd(&frc[atom_i].x, frc_record.x);
    atomicAdd(&frc[atom_i].y, frc_record.y);
    atomicAdd(&frc[atom_i].z, frc_record.z);
  }
}

static __global__ void
LJ_Force_With_Atom_Virial_CUDA(const int atom_numbers, const ATOM_GROUP *nl,
                               const UINT_VECTOR_LJ_TYPE *uint_crd,
                               const VECTOR boxlength, const float *LJ_type_A,
                               const float *LJ_type_B, const float cutoff,
                               VECTOR *frc, float *atom_lj_virial) {
  int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (atom_i < atom_numbers) {
    ATOM_GROUP nl_i = nl[atom_i];
    int N = nl_i.atom_numbers;
    // int B = (unsigned int)ceilf((float)N / blockDim.y);
    int atom_j;
    int int_x;
    int int_y;
    int int_z;
    UINT_VECTOR_LJ_TYPE r1 = uint_crd[atom_i], r2;
    VECTOR dr;
    // float dr2;
    float dr_2;
    float dr_4;
    float dr_8;
    float dr_6;
    float frc_abs = 0.0f;
    float dr_abs;
    float dr_1;
    VECTOR frc_lin;
    VECTOR frc_record = {0.0f, 0.0f, 0.0f};

    // LJ维里
    float virial_lin = 0.0f;
    int x, y;
    int atom_pair_LJ_type;
    for (int j = threadIdx.y; j < N; j = j + blockDim.y) {
      atom_j = nl_i.atom_serial[j];
      r2 = uint_crd[atom_j];

      int_x = r2.uint_x - r1.uint_x;
      int_y = r2.uint_y - r1.uint_y;
      int_z = r2.uint_z - r1.uint_z;
      dr.x = boxlength.x * int_x;
      dr.y = boxlength.y * int_y;
      dr.z = boxlength.z * int_z;
      dr_abs = norm3df(dr.x, dr.y, dr.z);
      if (dr_abs < cutoff) {
        dr_1 = 1. / dr_abs;
        dr_2 = dr_1 * dr_1;
        dr_4 = dr_2 * dr_2;
        dr_8 = dr_4 * dr_4;
        // dr_14 = dr_8*dr_4*dr_2;
        dr_6 = dr_4 * dr_2;

        y = (r2.LJ_type - r1.LJ_type);
        x = y >> 31;
        y = (y ^ x) - x;
        x = r2.LJ_type + r1.LJ_type;
        r2.LJ_type = (x + y) >> 1;
        x = (x - y) >> 1;
        atom_pair_LJ_type = (r2.LJ_type * (r2.LJ_type + 1) >> 1) + x;

        // frc_abs = -LJ_type_A[atom_pair_LJ_type] * dr_14
        //      + LJ_type_B[atom_pair_LJ_type] * dr_8;

        frc_abs = (-LJ_type_A[atom_pair_LJ_type] * dr_6 +
                   LJ_type_B[atom_pair_LJ_type]) *
                  dr_8;

        // LJ的dU/dr*r,frc_abs=dU/dr/r
        virial_lin = virial_lin - frc_abs * dr_abs * dr_abs;

        //两种力的绝对值
        frc_abs = frc_abs;

        frc_lin.x = frc_abs * dr.x;
        frc_lin.y = frc_abs * dr.y;
        frc_lin.z = frc_abs * dr.z;

        frc_record.x = frc_record.x + frc_lin.x;
        frc_record.y = frc_record.y + frc_lin.y;
        frc_record.z = frc_record.z + frc_lin.z;

        atomicAdd(&frc[atom_j].x, -frc_lin.x);
        atomicAdd(&frc[atom_j].y, -frc_lin.y);
        atomicAdd(&frc[atom_j].z, -frc_lin.z);
      }
    } // atom_j cycle
    atomicAdd(&frc[atom_i].x, frc_record.x);
    atomicAdd(&frc[atom_i].y, frc_record.y);
    atomicAdd(&frc[atom_i].z, frc_record.z);

    atomicAdd(&atom_lj_virial[atom_i], virial_lin);
  }
}

static __global__ void
LJ_Force_With_Atom_Energy_CUDA(const int atom_numbers, const ATOM_GROUP *nl,
                               const UINT_VECTOR_LJ_TYPE *uint_crd,
                               const VECTOR boxlength, const float *LJ_type_A,
                               const float *LJ_type_B, const float cutoff,
                               VECTOR *frc, float *atom_energy) {
  int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (atom_i < atom_numbers) {
    ATOM_GROUP nl_i = nl[atom_i];
    int N = nl_i.atom_numbers;
    // int B = (unsigned int)ceilf((float)N / blockDim.y);
    int atom_j;
    int int_x;
    int int_y;
    int int_z;
    UINT_VECTOR_LJ_TYPE r1 = uint_crd[atom_i], r2;
    VECTOR dr;
    // float dr2;
    float dr_2;
    float dr_4;
    float dr_8;
    float dr_6;
    float frc_abs = 0.0f;
    float dr_abs;
    float dr_1;
    VECTOR frc_lin;
    VECTOR frc_record = {0.0f, 0.0f, 0.0f};

    // lj能量
    float ene_lin = 0.0f;
    int x, y;
    int atom_pair_LJ_type;
    for (int j = threadIdx.y; j < N; j = j + blockDim.y) {
      atom_j = nl_i.atom_serial[j];
      r2 = uint_crd[atom_j];

      int_x = r2.uint_x - r1.uint_x;
      int_y = r2.uint_y - r1.uint_y;
      int_z = r2.uint_z - r1.uint_z;
      dr.x = boxlength.x * int_x;
      dr.y = boxlength.y * int_y;
      dr.z = boxlength.z * int_z;
      dr_abs = norm3df(dr.x, dr.y, dr.z);
      if (dr_abs < cutoff) {
        dr_1 = 1. / dr_abs;
        dr_2 = dr_1 * dr_1;
        dr_4 = dr_2 * dr_2;
        dr_8 = dr_4 * dr_4;
        // dr_14 = dr_8*dr_4*dr_2;
        dr_6 = dr_4 * dr_2;

        y = (r2.LJ_type - r1.LJ_type);
        x = y >> 31;
        y = (y ^ x) - x;
        x = r2.LJ_type + r1.LJ_type;
        r2.LJ_type = (x + y) >> 1;
        x = (x - y) >> 1;
        atom_pair_LJ_type = (r2.LJ_type * (r2.LJ_type + 1) >> 1) + x;

        // frc_abs = -LJ_type_A[atom_pair_LJ_type] * dr_14
        //    + LJ_type_B[atom_pair_LJ_type] * dr_8;

        frc_abs = (-LJ_type_A[atom_pair_LJ_type] * dr_6 +
                   LJ_type_B[atom_pair_LJ_type]) *
                  dr_8;

        //能量
        ene_lin = ene_lin + (0.083333333 * LJ_type_A[atom_pair_LJ_type] * dr_6 -
                             0.166666666 * LJ_type_B[atom_pair_LJ_type]) *
                                dr_6;

        //两种力的绝对值
        frc_abs = frc_abs;

        frc_lin.x = frc_abs * dr.x;
        frc_lin.y = frc_abs * dr.y;
        frc_lin.z = frc_abs * dr.z;

        frc_record.x = frc_record.x + frc_lin.x;
        frc_record.y = frc_record.y + frc_lin.y;
        frc_record.z = frc_record.z + frc_lin.z;

        atomicAdd(&frc[atom_j].x, -frc_lin.x);
        atomicAdd(&frc[atom_j].y, -frc_lin.y);
        atomicAdd(&frc[atom_j].z, -frc_lin.z);
      }
    } // atom_j cycle
    atomicAdd(&frc[atom_i].x, frc_record.x);
    atomicAdd(&frc[atom_i].y, frc_record.y);
    atomicAdd(&frc[atom_i].z, frc_record.z);

    atomicAdd(&atom_energy[atom_i], ene_lin);
  }
}

static __global__ void LJ_Force_With_Atom_Energy_And_Virial_CUDA(
    const int atom_numbers, const ATOM_GROUP *nl,
    const UINT_VECTOR_LJ_TYPE *uint_crd, const VECTOR boxlength,
    const float *LJ_type_A, const float *LJ_type_B, const float cutoff,
    VECTOR *frc, float *atom_energy, float *atom_lj_virial) {
  int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (atom_i < atom_numbers) {
    ATOM_GROUP nl_i = nl[atom_i];
    int N = nl_i.atom_numbers;
    // int B = (unsigned int)ceilf((float)N / blockDim.y);
    int atom_j;
    int int_x;
    int int_y;
    int int_z;
    UINT_VECTOR_LJ_TYPE r1 = uint_crd[atom_i], r2;
    VECTOR dr;
    // float dr2;
    float dr_2;
    float dr_4;
    float dr_8;
    float dr_6;
    float frc_abs = 0.0f;
    float dr_abs;
    float dr_1;
    VECTOR frc_lin;
    VECTOR frc_record = {0.0f, 0.0f, 0.0f};

    // LJ维里（未乘1/3/V）
    float virial_lin = 0.0f;
    // lj能量
    float ene_lin = 0.0f;
    int x, y;
    int atom_pair_LJ_type;
    for (int j = threadIdx.y; j < N; j = j + blockDim.y) {
      atom_j = nl_i.atom_serial[j];
      r2 = uint_crd[atom_j];

      int_x = r2.uint_x - r1.uint_x;
      int_y = r2.uint_y - r1.uint_y;
      int_z = r2.uint_z - r1.uint_z;
      dr.x = boxlength.x * int_x;
      dr.y = boxlength.y * int_y;
      dr.z = boxlength.z * int_z;
      dr_abs = norm3df(dr.x, dr.y, dr.z);
      if (dr_abs < cutoff) {
        dr_1 = 1. / dr_abs;
        dr_2 = dr_1 * dr_1;
        dr_4 = dr_2 * dr_2;
        dr_8 = dr_4 * dr_4;
        // dr_14 = dr_8*dr_4*dr_2;
        dr_6 = dr_4 * dr_2;

        y = (r2.LJ_type - r1.LJ_type);
        x = y >> 31;
        y = (y ^ x) - x;
        x = r2.LJ_type + r1.LJ_type;
        r2.LJ_type = (x + y) >> 1;
        x = (x - y) >> 1;
        atom_pair_LJ_type = (r2.LJ_type * (r2.LJ_type + 1) >> 1) + x;

        // frc_abs = -LJ_type_A[atom_pair_LJ_type] * dr_14
        //         + LJ_type_B[atom_pair_LJ_type] * dr_8;

        frc_abs = (-LJ_type_A[atom_pair_LJ_type] * dr_6 +
                   LJ_type_B[atom_pair_LJ_type]) *
                  dr_8;

        //能量
        ene_lin = ene_lin + (0.083333333 * LJ_type_A[atom_pair_LJ_type] * dr_6 -
                             0.166666666 * LJ_type_B[atom_pair_LJ_type]) *
                                dr_6;
        // LJ的dU/dr*r,frc_abs=dU/dr/r
        virial_lin = virial_lin - frc_abs * dr_abs * dr_abs;

        //两种力的绝对值
        frc_abs = frc_abs;

        frc_lin.x = frc_abs * dr.x;
        frc_lin.y = frc_abs * dr.y;
        frc_lin.z = frc_abs * dr.z;

        frc_record.x = frc_record.x + frc_lin.x;
        frc_record.y = frc_record.y + frc_lin.y;
        frc_record.z = frc_record.z + frc_lin.z;

        atomicAdd(&frc[atom_j].x, -frc_lin.x);
        atomicAdd(&frc[atom_j].y, -frc_lin.y);
        atomicAdd(&frc[atom_j].z, -frc_lin.z);
      }
    } // atom_j cycle
    atomicAdd(&frc[atom_i].x, frc_record.x);
    atomicAdd(&frc[atom_i].y, frc_record.y);
    atomicAdd(&frc[atom_i].z, frc_record.z);

    atomicAdd(&atom_lj_virial[atom_i], virial_lin);
    atomicAdd(&atom_energy[atom_i], ene_lin);
  }
}

__global__ void LJ_Energy_CUDA(const int atom_numbers, const ATOM_GROUP *nl,
                               const UINT_VECTOR_LJ_TYPE *uint_crd,
                               const VECTOR boxlength, const float *LJ_type_A,
                               const float *LJ_type_B,
                               const float cutoff_square, float *lj_ene) {
  int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (atom_i < atom_numbers) {
    ATOM_GROUP nl_i = nl[atom_i];
    int N = nl_i.atom_numbers;
    int atom_j;
    int int_x;
    int int_y;
    int int_z;
    UINT_VECTOR_LJ_TYPE r1 = uint_crd[atom_i], r2;
    VECTOR dr;
    float dr2;
    float dr_2;
    float dr_4;
    float dr_6;
    float ene_lin = 0.;

    int x, y;
    int atom_pair_LJ_type;
    for (int j = threadIdx.y; j < N; j = j + blockDim.y) {
      atom_j = nl_i.atom_serial[j];
      r2 = uint_crd[atom_j];

      int_x = r2.uint_x - r1.uint_x;
      int_y = r2.uint_y - r1.uint_y;
      int_z = r2.uint_z - r1.uint_z;
      dr.x = boxlength.x * int_x;
      dr.y = boxlength.y * int_y;
      dr.z = boxlength.z * int_z;

      dr2 = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;
      if (dr2 < cutoff_square) {
        dr_2 = 1. / dr2;
        dr_4 = dr_2 * dr_2;
        dr_6 = dr_4 * dr_2;

        y = (r2.LJ_type - r1.LJ_type);
        x = y >> 31;
        y = (y ^ x) - x;
        x = r2.LJ_type + r1.LJ_type;
        r2.LJ_type = (x + y) >> 1;
        x = (x - y) >> 1;
        atom_pair_LJ_type = (r2.LJ_type * (r2.LJ_type + 1) >> 1) + x;

        dr_2 = (0.083333333 * LJ_type_A[atom_pair_LJ_type] * dr_6 -
                0.166666666 * LJ_type_B[atom_pair_LJ_type]) *
               dr_6; // LJ的A,B系数已经乘以12和6因此要反乘
        ene_lin = ene_lin + dr_2;
      }
    } // atom_j cycle
    atomicAdd(&lj_ene[atom_i], ene_lin);
  }
}

void LENNARD_JONES_INFORMATION::LJ_Malloc() {
  Malloc_Safely((void **)&h_LJ_energy_sum, sizeof(float));
  Malloc_Safely((void **)&h_LJ_energy_atom, sizeof(float) * atom_numbers);
  Malloc_Safely((void **)&h_atom_LJ_type, sizeof(int) * atom_numbers);
  Malloc_Safely((void **)&h_LJ_A, sizeof(float) * pair_type_numbers);
  Malloc_Safely((void **)&h_LJ_B, sizeof(float) * pair_type_numbers);

  Cuda_Malloc_Safely((void **)&d_LJ_energy_sum, sizeof(float));
  Cuda_Malloc_Safely((void **)&d_LJ_energy_atom, sizeof(float) * atom_numbers);
  Cuda_Malloc_Safely((void **)&d_atom_LJ_type, sizeof(int) * atom_numbers);
  Cuda_Malloc_Safely((void **)&d_LJ_A, sizeof(float) * pair_type_numbers);
  Cuda_Malloc_Safely((void **)&d_LJ_B, sizeof(float) * pair_type_numbers);
}

static __global__ void Total_C6_Get(int atom_numbers, int *atom_lj_type,
                                    float *d_lj_b, float *d_factor) {
  int i, j;
  float temp_sum = 0;
  int x, y;
  int itype, jtype, atom_pair_LJ_type;
  for (i = blockIdx.x * blockDim.x + threadIdx.x; i < atom_numbers;
       i += gridDim.x * blockDim.x) {
    itype = atom_lj_type[i];
    for (j = blockIdx.y * blockDim.y + threadIdx.y; j < atom_numbers;
         j += gridDim.y * blockDim.y) {
      jtype = atom_lj_type[j];
      y = (jtype - itype);
      x = y >> 31;
      y = (y ^ x) - x;
      x = jtype + itype;
      jtype = (x + y) >> 1;
      x = (x - y) >> 1;
      atom_pair_LJ_type = (jtype * (jtype + 1) >> 1) + x;
      temp_sum += d_lj_b[atom_pair_LJ_type];
    }
  }
  atomicAdd(d_factor, temp_sum);
}

void LENNARD_JONES_INFORMATION::Initial(CONTROLLER *controller, float cutoff,
                                        VECTOR box_length,
                                        const char *module_name) {
  if (module_name == NULL) {
    strcpy(this->module_name, "LJ");
  } else {
    strcpy(this->module_name, module_name);
  }
  controller[0].printf("START INITIALIZING LENNADR JONES INFORMATION:\n");
  if (controller[0].Command_Exist(this->module_name, "in_file")) {
    FILE *fp = NULL;
    Open_File_Safely(&fp, controller[0].Command(this->module_name, "in_file"),
                     "r");

    int scanf_ret = fscanf(fp, "%d %d", &atom_numbers, &atom_type_numbers);
    controller[0].printf("    atom_numbers is %d\n", atom_numbers);
    controller[0].printf("    atom_LJ_type_number is %d\n", atom_type_numbers);
    pair_type_numbers = atom_type_numbers * (atom_type_numbers + 1) / 2;
    LJ_Malloc();

    for (int i = 0; i < pair_type_numbers; i++) {
      scanf_ret = fscanf(fp, "%f", h_LJ_A + i);
      h_LJ_A[i] *= 12.0f;
    }
    for (int i = 0; i < pair_type_numbers; i++) {
      scanf_ret = fscanf(fp, "%f", h_LJ_B + i);
      h_LJ_B[i] *= 6.0f;
    }
    for (int i = 0; i < atom_numbers; i++) {
      scanf_ret = fscanf(fp, "%d", h_atom_LJ_type + i);
    }
    fclose(fp);
    Parameter_Host_To_Device();
    is_initialized = 1;
  } else if (controller[0].Command_Exist("amber_parm7")) {
    Initial_From_AMBER_Parm(controller[0].Command("amber_parm7"),
                            controller[0]);
  }
  if (is_initialized) {
    this->cutoff = cutoff;
    this->uint_dr_to_dr_cof = 1.0f / CONSTANT_UINT_MAX_FLOAT * box_length;
    Cuda_Malloc_Safely((void **)&uint_crd_with_LJ,
                       sizeof(UINT_VECTOR_LJ_TYPE) * atom_numbers);
    Copy_LJ_Type_To_New_Crd<<<ceilf((float)this->atom_numbers / 32), 32>>>(
        atom_numbers, uint_crd_with_LJ, d_atom_LJ_type);
    controller[0].printf("    Start initializing long range LJ correction\n");
    long_range_factor = 0;
    float *d_factor = NULL;
    Cuda_Malloc_Safely((void **)&d_factor, sizeof(float));
    Reset_List(d_factor, 0.0f, 1, 1);
    Total_C6_Get<<<{4, 4}, {32, 32}>>>(atom_numbers, d_atom_LJ_type, d_LJ_B,
                                       d_factor);
    cudaMemcpy(&long_range_factor, d_factor, sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaFree(d_factor);

    long_range_factor *=
        -2.0f / 3.0f * CONSTANT_Pi / cutoff / cutoff / cutoff / 6.0f;
    this->volume = box_length.x * box_length.y * box_length.z;
    controller[0].printf("        long range correction factor is: %e\n",
                         long_range_factor);
    controller[0].printf("    End initializing long range LJ correction\n");
  }
  if (is_initialized && !is_controller_printf_initialized) {
    controller[0].Step_Print_Initial(this->module_name, "%.2f");
    is_controller_printf_initialized = 1;
    controller[0].printf("    structure last modify date is %d\n",
                         last_modify_date);
  }
  controller[0].printf("END INITIALIZING LENNADR JONES INFORMATION\n\n");
}

void LENNARD_JONES_INFORMATION::Clear() {
  if (is_initialized) {
    is_initialized = 0;

    free(h_atom_LJ_type);
    cudaFree(d_atom_LJ_type);

    free(h_LJ_A);
    free(h_LJ_B);
    cudaFree(d_LJ_A);
    cudaFree(d_LJ_B);

    free(h_LJ_energy_atom);
    cudaFree(d_LJ_energy_atom);
    cudaFree(d_LJ_energy_sum);

    cudaFree(uint_crd_with_LJ);

    h_atom_LJ_type = NULL;
    d_atom_LJ_type = NULL;

    h_LJ_A = NULL;
    h_LJ_B = NULL;
    d_LJ_A = NULL;
    d_LJ_B = NULL;

    h_LJ_energy_atom = NULL;
    d_LJ_energy_atom = NULL;
    d_LJ_energy_sum = NULL;

    uint_crd_with_LJ = NULL;
  }
}

void LENNARD_JONES_INFORMATION::Long_Range_Correction(int need_pressure,
                                                      float *d_virial,
                                                      int need_potential,
                                                      float *d_potential) {
  if (is_initialized) {
    if (need_pressure > 0) {
      device_add<<<1, 1>>>(d_virial, long_range_factor * 6.0f / volume);
    }
    if (need_potential > 0) {
      device_add<<<1, 1>>>(d_potential, long_range_factor / volume);
    }
  }
}

void LENNARD_JONES_INFORMATION::Long_Range_Correction(float volume) {
  if (is_initialized) {
    device_add<<<1, 1>>>(d_LJ_energy_sum, long_range_factor / volume);
  }
}

void LENNARD_JONES_INFORMATION::Initial_From_AMBER_Parm(const char *file_name,
                                                        CONTROLLER controller) {
  FILE *parm = NULL;
  Open_File_Safely(&parm, file_name, "r");
  controller.printf("    Start reading LJ information from AMBER file:\n");

  while (true) {
    char temps[CHAR_LENGTH_MAX];
    char temp_first_str[CHAR_LENGTH_MAX];
    char temp_second_str[CHAR_LENGTH_MAX];
    if (!fgets(temps, CHAR_LENGTH_MAX, parm)) {
      break;
    }
    if (sscanf(temps, "%s %s", temp_first_str, temp_second_str) != 2) {
      continue;
    }
    if (strcmp(temp_first_str, "%FLAG") == 0 &&
        strcmp(temp_second_str, "POINTERS") == 0) {
      char *get_ret = fgets(temps, CHAR_LENGTH_MAX, parm);

      int scanf_ret = fscanf(parm, "%d\n", &atom_numbers);
      controller.printf("        atom_numbers is %d\n", atom_numbers);
      scanf_ret = fscanf(parm, "%d\n", &atom_type_numbers);
      controller.printf("        atom_LJ_type_number is %d\n",
                        atom_type_numbers);
      pair_type_numbers = atom_type_numbers * (atom_type_numbers + 1) / 2;

      LJ_Malloc();
    }
    if (strcmp(temp_first_str, "%FLAG") == 0 &&
        strcmp(temp_second_str, "ATOM_TYPE_INDEX") == 0) {
      char *get_ret = fgets(temps, CHAR_LENGTH_MAX, parm);
      printf("     read atom LJ type index\n");
      int atomljtype;
      for (int i = 0; i < atom_numbers; i = i + 1) {
        int scanf_ret = fscanf(parm, "%d\n", &atomljtype);
        h_atom_LJ_type[i] = atomljtype - 1;
      }
    }
    if (strcmp(temp_first_str, "%FLAG") == 0 &&
        strcmp(temp_second_str, "LENNARD_JONES_ACOEF") == 0) {
      char *get_ret = fgets(temps, CHAR_LENGTH_MAX, parm);
      printf("     read atom LJ A\n");
      double lin;
      for (int i = 0; i < pair_type_numbers; i = i + 1) {
        int scanf_ret = fscanf(parm, "%lf\n", &lin);
        h_LJ_A[i] = (float)12. * lin;
      }
    }
    if (strcmp(temp_first_str, "%FLAG") == 0 &&
        strcmp(temp_second_str, "LENNARD_JONES_BCOEF") == 0) {
      char *get_ret = fgets(temps, CHAR_LENGTH_MAX, parm);
      printf("     read atom LJ B\n");
      double lin;
      for (int i = 0; i < pair_type_numbers; i = i + 1) {
        int scanf_ret = fscanf(parm, "%lf\n", &lin);
        h_LJ_B[i] = (float)6. * lin;
      }
    }
  }
  controller.printf("    End reading LJ information from AMBER file:\n");
  fclose(parm);
  is_initialized = 1;
  Parameter_Host_To_Device();
}

void LENNARD_JONES_INFORMATION::Parameter_Host_To_Device() {
  cudaMemcpy(d_LJ_B, h_LJ_B, sizeof(float) * pair_type_numbers,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_LJ_A, h_LJ_A, sizeof(float) * pair_type_numbers,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_atom_LJ_type, h_atom_LJ_type, sizeof(int) * atom_numbers,
             cudaMemcpyHostToDevice);
}

void LENNARD_JONES_INFORMATION::LJ_Force(const UINT_VECTOR_LJ_TYPE *uint_crd,
                                         const VECTOR scaler, VECTOR *frc,
                                         const ATOM_GROUP *nl,
                                         const float cutoff) {
  if (is_initialized)
    LJ_Force_CUDA<<<(unsigned int)ceilf((float)atom_numbers / thread_LJ.x),
                    thread_LJ>>>(atom_numbers, nl, uint_crd, scaler, d_LJ_A,
                                 d_LJ_B, cutoff, frc);
}

void LENNARD_JONES_INFORMATION::LJ_Force_With_PME_Direct_Force(
    const int atom_numbers, const UINT_VECTOR_LJ_TYPE *uint_crd,
    const VECTOR scaler, VECTOR *frc, const ATOM_GROUP *nl, const float cutoff,
    const float pme_beta) {
  if (is_initialized)
    LJ_Force_With_Direct_CF_CUDA<<<
        {1, (unsigned int)ceilf((float)atom_numbers / thread_LJ.y)},
        thread_LJ>>>(atom_numbers, nl, uint_crd, scaler, d_LJ_A, d_LJ_B, cutoff,
                     frc, pme_beta, TWO_DIVIDED_BY_SQRT_PI);
}
void LENNARD_JONES_INFORMATION::LJ_PME_Direct_Force_With_Atom_Energy(
    const int atom_numbers, const UINT_VECTOR_LJ_TYPE *uint_crd,
    const VECTOR scaler, VECTOR *frc, const ATOM_GROUP *nl, const float cutoff,
    const float pme_beta, float *atom_energy) {
  if (is_initialized)
    LJ_Direct_CF_Force_With_Atom_Energy_CUDA<<<
        {1, (unsigned int)ceilf((float)atom_numbers / thread_LJ.y)},
        thread_LJ>>>(atom_numbers, nl, uint_crd, scaler, d_LJ_A, d_LJ_B, cutoff,
                     frc, pme_beta, TWO_DIVIDED_BY_SQRT_PI, atom_energy);
}
void LENNARD_JONES_INFORMATION::LJ_PME_Direct_Force_With_Atom_Energy_And_Virial(
    const int atom_numbers, const UNSIGNED_INT_VECTOR *uint_crd,
    const float *charge, VECTOR *frc, const ATOM_GROUP *nl,
    const float pme_beta, const int need_atom_energy, float *atom_energy,
    const int need_virial, float *atom_lj_virial,
    float *atom_direct_pme_energy) {
  if (is_initialized) {
    Copy_Crd_And_Charge_To_New_Crd<<<
        (unsigned int)ceilf((float)atom_numbers / 1024), 1024>>>(
        atom_numbers, uint_crd, uint_crd_with_LJ, charge);
    if (!need_atom_energy > 0 && !need_virial > 0) {
      LJ_Force_With_PME_Direct_Force(atom_numbers, uint_crd_with_LJ,
                                     uint_dr_to_dr_cof, frc, nl, cutoff,
                                     pme_beta);
    } else if (need_atom_energy > 0 && !need_virial > 0) {
      LJ_PME_Direct_Force_With_Atom_Energy(atom_numbers, uint_crd_with_LJ,
                                           uint_dr_to_dr_cof, frc, nl, cutoff,
                                           pme_beta, atom_energy);
    } else if (!need_atom_energy > 0 && need_virial > 0) {
      Reset_List(atom_direct_pme_energy, 0.0f, atom_numbers, 1024);
      LJ_Direct_CF_Force_With_LJ_Virial_Direct_CF_Energy_CUDA<<<
          {1, (unsigned int)ceilf((float)atom_numbers / thread_LJ.y)},
          thread_LJ>>>(atom_numbers, nl, uint_crd_with_LJ, uint_dr_to_dr_cof,
                       d_LJ_A, d_LJ_B, cutoff, frc, pme_beta,
                       TWO_DIVIDED_BY_SQRT_PI, atom_lj_virial,
                       atom_direct_pme_energy);
    } else {
      Reset_List(atom_direct_pme_energy, 0.0f, atom_numbers, 1024);
      LJ_Direct_CF_Force_With_Atom_Energy_And_LJ_Virial_Direct_CF_Energy_CUDA<<<
          {1, (unsigned int)ceilf((float)atom_numbers / thread_LJ.y)},
          thread_LJ>>>(atom_numbers, nl, uint_crd_with_LJ, uint_dr_to_dr_cof,
                       d_LJ_A, d_LJ_B, cutoff, frc, pme_beta,
                       TWO_DIVIDED_BY_SQRT_PI, atom_energy, atom_lj_virial,
                       atom_direct_pme_energy);
    }
  }
}

void LENNARD_JONES_INFORMATION::LJ_Force_With_Atom_Energy_And_Virial(
    const int atom_numbers, const UINT_VECTOR_LJ_TYPE *uint_crd,
    const VECTOR scaler, VECTOR *frc, const ATOM_GROUP *nl, const float cutoff,
    int need_atom_energy, float *atom_energy, int need_virial,
    float *atom_lj_virial, float *virial) {
  if (is_initialized) {
    if (!need_atom_energy && !need_virial) {
      LJ_Force_CUDA<<<(unsigned int)ceilf((float)atom_numbers / thread_LJ.x),
                      thread_LJ>>>(atom_numbers, nl, uint_crd, scaler, d_LJ_A,
                                   d_LJ_B, cutoff, frc);
    } else if (need_atom_energy && !need_virial) {
      LJ_Force_With_Atom_Energy_CUDA<<<
          (unsigned int)ceilf((float)atom_numbers / thread_LJ.x), thread_LJ>>>(
          atom_numbers, nl, uint_crd, scaler, d_LJ_A, d_LJ_B, cutoff, frc,
          atom_energy);
    } else if (!need_atom_energy && need_virial) {
      LJ_Force_With_Atom_Virial_CUDA<<<
          (unsigned int)ceilf((float)atom_numbers / thread_LJ.x), thread_LJ>>>(
          atom_numbers, nl, uint_crd, scaler, d_LJ_A, d_LJ_B, cutoff, frc,
          atom_lj_virial);
    } else {
      LJ_Force_With_Atom_Energy_And_Virial_CUDA<<<
          (unsigned int)ceilf((float)atom_numbers / thread_LJ.x), thread_LJ>>>(
          atom_numbers, nl, uint_crd, scaler, d_LJ_A, d_LJ_B, cutoff, frc,
          atom_energy, atom_lj_virial);
    }
  }
}

void LENNARD_JONES_INFORMATION::LJ_Energy(const int atom_numbers,
                                          const UINT_VECTOR_LJ_TYPE *uint_crd,
                                          const VECTOR scaler,
                                          const ATOM_GROUP *nl,
                                          const float cutoff_square,
                                          float *d_LJ_energy_atom) {
  if (is_initialized)
    LJ_Energy_CUDA<<<(unsigned int)ceilf((float)atom_numbers / thread_LJ.x),
                     thread_LJ>>>(atom_numbers, nl, uint_crd, scaler, d_LJ_A,
                                  d_LJ_B, cutoff_square, d_LJ_energy_atom);
}

float LENNARD_JONES_INFORMATION::Get_Energy(const UNSIGNED_INT_VECTOR *uint_crd,
                                            const ATOM_GROUP *nl,
                                            int is_download) {
  if (is_initialized) {
    Copy_Crd_To_New_Crd<<<(unsigned int)ceilf((float)atom_numbers / 32), 32>>>(
        atom_numbers, uint_crd, uint_crd_with_LJ);
    Reset_List(d_LJ_energy_atom, 0., atom_numbers, 1024);
    LJ_Energy_CUDA<<<(unsigned int)ceilf((float)atom_numbers / thread_LJ.x),
                     thread_LJ>>>(atom_numbers, nl, uint_crd_with_LJ,
                                  uint_dr_to_dr_cof, d_LJ_A, d_LJ_B,
                                  cutoff * cutoff, d_LJ_energy_atom);
    Sum_Of_List(d_LJ_energy_atom, d_LJ_energy_sum, atom_numbers);

    device_add<<<1, 1>>>(d_LJ_energy_sum, long_range_factor / volume);

    if (is_download) {
      cudaMemcpy(&h_LJ_energy_sum, this->d_LJ_energy_sum, sizeof(float),
                 cudaMemcpyDeviceToHost);
      return h_LJ_energy_sum;
    } else {
      return 0;
    }
  }
  return NAN;
}

void LENNARD_JONES_INFORMATION::LJ_Energy(const UNSIGNED_INT_VECTOR *uint_crd,
                                          const ATOM_GROUP *nl) {
  if (is_initialized) {
    Copy_Crd_To_New_Crd<<<(unsigned int)ceilf((float)atom_numbers / 32), 32>>>(
        atom_numbers, uint_crd, uint_crd_with_LJ);
    Reset_List(d_LJ_energy_atom, 0., atom_numbers, 1024);
    LJ_Energy_CUDA<<<(unsigned int)ceilf((float)atom_numbers / thread_LJ.x),
                     thread_LJ>>>(atom_numbers, nl, uint_crd_with_LJ,
                                  uint_dr_to_dr_cof, d_LJ_A, d_LJ_B,
                                  cutoff * cutoff, d_LJ_energy_atom);
    Sum_Of_List(d_LJ_energy_atom, d_LJ_energy_sum, atom_numbers);
  }
}

void LENNARD_JONES_INFORMATION::Update_Volume(VECTOR box_length) {
  if (!is_initialized)
    return;
  this->uint_dr_to_dr_cof = 1.0f / CONSTANT_UINT_MAX_FLOAT * box_length;
  this->volume = box_length.x * box_length.y * box_length.z;
}

void LENNARD_JONES_INFORMATION::Energy_Device_To_Host() {
  cudaMemcpy(&h_LJ_energy_sum, d_LJ_energy_sum, sizeof(float),
             cudaMemcpyDeviceToHost);
}

__global__ void LJ_Force_With_FGM_Direct_Force_CUDA(
    const int atom_numbers, const ATOM_GROUP *nl,
    const UINT_VECTOR_LJ_TYPE *uint_crd, const VECTOR boxlength,
    const float *LJ_type_A, const float *LJ_type_B, const float cutoff,
    VECTOR *frc, const float CUBIC_R_INVERSE) {
  int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (atom_i < atom_numbers) {
    ATOM_GROUP nl_i = nl[atom_i];
    int N = nl_i.atom_numbers;
    // int B = (unsigned int)ceilf((float)N / blockDim.y);
    int atom_j;
    int int_x;
    int int_y;
    int int_z;
    UINT_VECTOR_LJ_TYPE r1 = uint_crd[atom_i], r2;
    VECTOR dr;
    // float dr2;
    float dr_2;
    float dr_4;
    float dr_8;
    float dr_6;
    float frc_abs = 0.;
    VECTOR frc_lin;
    VECTOR frc_record = {0., 0., 0.};

    // CF
    float charge_i = r1.charge; // r1.charge;
    float charge_j;
    float dr_abs;
    float dr_1;
    float frc_cf_abs;
    //

    int x, y;
    int atom_pair_LJ_type;
    for (int j = threadIdx.y; j < N; j = j + blockDim.y) {
      atom_j = nl_i.atom_serial[j];
      r2 = uint_crd[atom_j];
      // CF
      charge_j = r2.charge;

      int_x = r2.uint_x - r1.uint_x;
      int_y = r2.uint_y - r1.uint_y;
      int_z = r2.uint_z - r1.uint_z;
      dr.x = boxlength.x * int_x;
      dr.y = boxlength.y * int_y;
      dr.z = boxlength.z * int_z;
      dr_abs = norm3df(dr.x, dr.y, dr.z);
      if (dr_abs < cutoff) {
        dr_1 = 1. / dr_abs;
        dr_2 = dr_1 * dr_1;
        dr_4 = dr_2 * dr_2;
        dr_8 = dr_4 * dr_4;
        // dr_14 = dr_8*dr_4*dr_2;
        dr_6 = dr_4 * dr_2;

        y = (r2.LJ_type - r1.LJ_type);
        x = y >> 31;
        y = (y ^ x) - x;
        x = r2.LJ_type + r1.LJ_type;
        r2.LJ_type = (x + y) >> 1;
        x = (x - y) >> 1;
        atom_pair_LJ_type = (r2.LJ_type * (r2.LJ_type + 1) >> 1) + x;

        frc_abs = (-LJ_type_A[atom_pair_LJ_type] * dr_6 +
                   LJ_type_B[atom_pair_LJ_type]) *
                  dr_8;
        // CF
        // charge_j = charge[atom_j];
        // dr_abs = sqrtf(dr2);
        // sqrt_pi= 2/sqrt(3.141592654);
        frc_cf_abs = (dr_2 * dr_1 - CUBIC_R_INVERSE);
        frc_cf_abs = charge_i * charge_j * frc_cf_abs;

        frc_abs = frc_abs - frc_cf_abs;
        // frc_abs = frc_abs + dr_2*sqrtf(dr_2) * 7.72261074E-01
        // * 7.72261074E-01;//charge

        frc_lin.x = frc_abs * dr.x;
        frc_lin.y = frc_abs * dr.y;
        frc_lin.z = frc_abs * dr.z;

        frc_record.x = frc_record.x + frc_lin.x;
        frc_record.y = frc_record.y + frc_lin.y;
        frc_record.z = frc_record.z + frc_lin.z;

        atomicAdd(&frc[atom_j].x, -frc_lin.x);
        atomicAdd(&frc[atom_j].y, -frc_lin.y);
        atomicAdd(&frc[atom_j].z, -frc_lin.z);
      }
    } // atom_j cycle
    atomicAdd(&frc[atom_i].x, frc_record.x);
    atomicAdd(&frc[atom_i].y, frc_record.y);
    atomicAdd(&frc[atom_i].z, frc_record.z);
  }
}
