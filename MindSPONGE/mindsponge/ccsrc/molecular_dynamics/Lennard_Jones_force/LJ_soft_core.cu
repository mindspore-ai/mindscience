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

#include "LJ_soft_core.cuh"

#define TWO_DIVIDED_BY_SQRT_PI 1.1283791670218446
#define ONE_DIVIDED_BY_3_SQRT_PI 0.18806319451591877

float Get_Beta_LJ(float cutoff, float tolerance) {
  float beta, low, high, tempf;
  int ilow, ihigh;

  high = 1.0;
  ihigh = 1;

  while (1) {
    tempf = erfc(high * cutoff) / cutoff;
    if (tempf <= tolerance)
      break;
    high *= 2;
    ihigh++;
  }

  ihigh += 50;
  low = 0.0;
  for (ilow = 1; ilow < ihigh; ilow++) {
    beta = (low + high) / 2;
    tempf = erfc(beta * cutoff) / cutoff;
    if (tempf >= tolerance)
      low = beta;
    else
      high = beta;
  }
  return beta;
}

__device__ __host__ VECTOR Get_Periodic_Displacement(
    const UINT_VECTOR_LJ_FEP_TYPE uvec_a, const UINT_VECTOR_LJ_FEP_TYPE uvec_b,
    const VECTOR scaler) {
  VECTOR dr;
  dr.x = ((int)(uvec_a.uint_x - uvec_b.uint_x)) * scaler.x;
  dr.y = ((int)(uvec_a.uint_y - uvec_b.uint_y)) * scaler.y;
  dr.z = ((int)(uvec_a.uint_z - uvec_b.uint_z)) * scaler.z;
  return dr;
}

__global__ void Copy_LJ_Type_And_Mask_To_New_Crd(
    const int atom_numbers, UINT_VECTOR_LJ_FEP_TYPE *new_crd,
    const int *LJ_type_A, const int *LJ_type_B, const int *mask) {
  int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (atom_i < atom_numbers) {
    new_crd[atom_i].LJ_type_A = LJ_type_A[atom_i];
    new_crd[atom_i].LJ_type_B = LJ_type_B[atom_i];
    new_crd[atom_i].mask = mask[atom_i];
  }
}

static __global__ void device_add(float *variable, const float adder) {
  variable[0] += adder;
}

__global__ void Copy_Crd_And_Charge_To_New_Crd(const int atom_numbers,
                                               const UNSIGNED_INT_VECTOR *crd,
                                               UINT_VECTOR_LJ_FEP_TYPE *new_crd,
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
                                    UINT_VECTOR_LJ_FEP_TYPE *new_crd) {
  int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (atom_i < atom_numbers) {
    new_crd[atom_i].uint_x = crd[atom_i].uint_x;
    new_crd[atom_i].uint_y = crd[atom_i].uint_y;
    new_crd[atom_i].uint_z = crd[atom_i].uint_z;
  }
}

static __global__ void Total_C6_Get(int atom_numbers, int *atom_lj_type_A,
                                    int *atom_lj_type_B, float *d_lj_Ab,
                                    float *d_lj_Bb, float *d_factor,
                                    const float lambda) {
  int i, j;
  float temp_sum = 0.0;
  int xA, yA, xB, yB;
  int itype_A, jtype_A, itype_B, jtype_B, atom_pair_LJ_type_A,
      atom_pair_LJ_type_B;
  float lambda_ = 1.0 - lambda;
  for (i = blockIdx.x * blockDim.x + threadIdx.x; i < atom_numbers;
       i += gridDim.x * blockDim.x) {
    itype_A = atom_lj_type_A[i];
    itype_B = atom_lj_type_B[i];
    for (j = blockIdx.y * blockDim.y + threadIdx.y; j < atom_numbers;
         j += gridDim.y * blockDim.y) {
      jtype_A = atom_lj_type_A[j];
      jtype_B = atom_lj_type_B[j];
      yA = (jtype_A - itype_A);
      xA = yA >> 31;
      yA = (yA ^ xA) - xA;
      xA = jtype_A + itype_A;
      jtype_A = (xA + yA) >> 1;
      xA = (xA - yA) >> 1;
      atom_pair_LJ_type_A = (jtype_A * (jtype_A + 1) >> 1) + xA;

      yB = (jtype_B - itype_B);
      xB = yB >> 31;
      yB = (yB ^ xB) - xB;
      xB = jtype_B + itype_B;
      jtype_B = (xB + yB) >> 1;
      xB = (xB - yB) >> 1;
      atom_pair_LJ_type_B = (jtype_B * (jtype_B + 1) >> 1) + xB;

      temp_sum += lambda_ * d_lj_Ab[atom_pair_LJ_type_A];
      temp_sum += lambda * d_lj_Bb[atom_pair_LJ_type_B];
    }
  }
  atomicAdd(d_factor, temp_sum);
}

static __global__ void Total_C6_B_A_Get(int atom_numbers, int *atom_lj_type_A,
                                        int *atom_lj_type_B, float *d_lj_Ab,
                                        float *d_lj_Bb, float *d_factor) {
  int i, j;
  float temp_sum = 0.0;
  int xA, yA, xB, yB;
  int itype_A, jtype_A, itype_B, jtype_B, atom_pair_LJ_type_A,
      atom_pair_LJ_type_B;
  for (i = blockIdx.x * blockDim.x + threadIdx.x; i < atom_numbers;
       i += gridDim.x * blockDim.x) {
    itype_A = atom_lj_type_A[i];
    itype_B = atom_lj_type_B[i];
    for (j = blockIdx.y * blockDim.y + threadIdx.y; j < atom_numbers;
         j += gridDim.y * blockDim.y) {
      jtype_A = atom_lj_type_A[j];
      jtype_B = atom_lj_type_B[j];
      yA = (jtype_A - itype_A);
      xA = yA >> 31;
      yA = (yA ^ xA) - xA;
      xA = jtype_A + itype_A;
      jtype_A = (xA + yA) >> 1;
      xA = (xA - yA) >> 1;
      atom_pair_LJ_type_A = (jtype_A * (jtype_A + 1) >> 1) + xA;

      yB = (jtype_B - itype_B);
      xB = yB >> 31;
      yB = (yB ^ xB) - xB;
      xB = jtype_B + itype_B;
      jtype_B = (xB + yB) >> 1;
      xB = (xB - yB) >> 1;
      atom_pair_LJ_type_B = (jtype_B * (jtype_B + 1) >> 1) + xB;

      temp_sum += d_lj_Bb[atom_pair_LJ_type_B] - d_lj_Ab[atom_pair_LJ_type_A];
    }
  }
  atomicAdd(d_factor, temp_sum);
}

static __global__ void LJ_Soft_Core_Force_With_Direct_CF_CUDA(
    const int atom_numbers, const ATOM_GROUP *nl,
    const UINT_VECTOR_LJ_FEP_TYPE *uint_crd, const VECTOR boxlength,
    const float *LJ_type_AA, const float *LJ_type_AB, const float *LJ_type_BA,
    const float *LJ_type_BB, const float cutoff, VECTOR *frc,
    const float pme_beta, const float sqrt_pi, const float lambda,
    const float alpha_lambda_p, const float alpha_lambda_p_,
    const float input_sigma_6, const float input_sigma_6_min) {
  int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
  float lambda_ = 1.0 - lambda;
  if (atom_i < atom_numbers) {
    ATOM_GROUP nl_i = nl[atom_i];
    int N = nl_i.atom_numbers;
    int atom_j;
    int int_x;
    int int_y;
    int int_z;
    UINT_VECTOR_LJ_FEP_TYPE r1 = uint_crd[atom_i], r2;
    VECTOR dr;
    float dr2, dr4, dr6;
    float dr_sc_A6, dr_sc_B6;
    float dr_sc_A, dr_sc_B;
    float dr_sc_A12, dr_sc_B12;
    float dr_2;
    float dr_4;
    float dr_8;
    float dr_6;
    float frc_abs = 0.;
    float AAij, ABij, BAij, BBij;
    float sigma_Aij, sigma_Bij;
    VECTOR frc_lin;
    VECTOR frc_record = {0., 0., 0.};

    // CF
    float charge_i = r1.charge; // r1.charge;
    float charge_j;
    float dr_abs;
    float dr_1;
    float beta_dr;
    float beta_dr_sc_A, beta_dr_sc_B;
    float frc_cf_abs;
    //

    int xA, yA, xB, yB;
    int atom_pair_LJ_type_A, atom_pair_LJ_type_B;

    int mask_i = r1.mask, mask_j;
    bool soft_core;
    for (int j = threadIdx.y; j < N; j = j + blockDim.y) {
      atom_j = nl_i.atom_serial[j];
      r2 = uint_crd[atom_j];
      // CF
      charge_j = r2.charge;
      mask_j = r2.mask;

      int_x = r2.uint_x - r1.uint_x;
      int_y = r2.uint_y - r1.uint_y;
      int_z = r2.uint_z - r1.uint_z;
      dr.x = boxlength.x * int_x;
      dr.y = boxlength.y * int_y;
      dr.z = boxlength.z * int_z;
      dr_abs = norm3df(dr.x, dr.y, dr.z);
      if (dr_abs < cutoff) {
        yA = (r2.LJ_type_A - r1.LJ_type_A);
        xA = yA >> 31;
        yA = (yA ^ xA) - xA;
        xA = r2.LJ_type_A + r1.LJ_type_A;
        r2.LJ_type_A = (xA + yA) >> 1;
        xA = (xA - yA) >> 1;
        atom_pair_LJ_type_A = (r2.LJ_type_A * (r2.LJ_type_A + 1) >> 1) + xA;
        AAij = LJ_type_AA[atom_pair_LJ_type_A];
        ABij = LJ_type_AB[atom_pair_LJ_type_A];

        yB = (r2.LJ_type_B - r1.LJ_type_B);
        xB = yB >> 31;
        yB = (yB ^ xB) - xB;
        xB = r2.LJ_type_B + r1.LJ_type_B;
        r2.LJ_type_B = (xB + yB) >> 1;
        xB = (xB - yB) >> 1;
        atom_pair_LJ_type_B = (r2.LJ_type_B * (r2.LJ_type_B + 1) >> 1) + xB;
        BAij = LJ_type_BA[atom_pair_LJ_type_B];
        BBij = LJ_type_BB[atom_pair_LJ_type_B];

        soft_core = (mask_i != mask_j) || (BAij > 1e-6 && AAij < 1e-6) ||
                    (BAij < 1e-6 && AAij > 1e-6);
        if (!soft_core) {
          dr_1 = 1. / dr_abs;
          dr_2 = dr_1 * dr_1;
          dr_4 = dr_2 * dr_2;
          dr_8 = dr_4 * dr_4;
          dr_6 = dr_4 * dr_2;
          frc_abs = (-(lambda_ * AAij + lambda * BAij) * dr_6 +
                     (lambda_ * ABij + lambda * BBij)) *
                    dr_8;

          beta_dr = pme_beta * dr_abs;
          frc_cf_abs =
              beta_dr * sqrt_pi * expf(-beta_dr * beta_dr) + erfcf(beta_dr);
          frc_cf_abs = frc_cf_abs * dr_2 * dr_1;
          frc_cf_abs = charge_i * charge_j * frc_cf_abs;

          frc_abs = frc_abs - frc_cf_abs;
        } else {
          dr2 = dr_abs * dr_abs;
          dr4 = dr2 * dr2;
          dr6 = dr4 * dr2;
          if (AAij < 1e-6 || ABij < 1e-6)
            sigma_Aij = input_sigma_6;
          else
            sigma_Aij = max(0.5 * AAij / ABij, input_sigma_6_min);
          if (BAij < 1e-6 || BBij < 1e-6)
            sigma_Bij = input_sigma_6;
          else
            sigma_Bij = max(0.5 * BAij / BBij, input_sigma_6_min);

          dr_sc_A6 = 1.0 / (dr6 + alpha_lambda_p * sigma_Aij);
          dr_sc_B6 = 1.0 / (dr6 + alpha_lambda_p_ * sigma_Bij);
          dr_sc_A12 = dr_sc_A6 * dr_sc_A6;
          dr_sc_B12 = dr_sc_B6 * dr_sc_B6;

          frc_abs = dr4 * (lambda_ * (-AAij * dr_sc_A6 + ABij) * dr_sc_A12 +
                           lambda * (-BAij * dr_sc_B6 + BBij) * dr_sc_B12);

          dr_sc_A = powf(dr_sc_A6, 1.0 / 6.0);
          dr_sc_B = powf(dr_sc_B6, 1.0 / 6.0);
          beta_dr_sc_A = pme_beta / dr_sc_A;
          beta_dr_sc_B = pme_beta / dr_sc_B;

          frc_cf_abs =
              dr4 *
              (lambda_ *
                   (expf(-beta_dr_sc_A * beta_dr_sc_A) * sqrt_pi * pme_beta +
                    erfcf(beta_dr_sc_A) * dr_sc_A) *
                   dr_sc_A6 +
               lambda *
                   (expf(-beta_dr_sc_B * beta_dr_sc_B) * sqrt_pi * pme_beta +
                    erfcf(beta_dr_sc_B) * dr_sc_B) *
                   dr_sc_B6);
          frc_cf_abs = frc_cf_abs * charge_i * charge_j;

          frc_abs = frc_abs - frc_cf_abs;
        }

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

static __global__ void LJ_Soft_Core_Direct_CF_Force_With_Atom_Energy_CUDA(
    const int atom_numbers, const ATOM_GROUP *nl,
    const UINT_VECTOR_LJ_FEP_TYPE *uint_crd, const VECTOR boxlength,
    const float *LJ_type_AA, const float *LJ_type_AB, const float *LJ_type_BA,
    const float *LJ_type_BB, const float cutoff, VECTOR *frc,
    const float pme_beta, const float sqrt_pi, float *atom_energy,
    const float lambda, const float alpha_lambda_p, const float alpha_lambda_p_,
    const float input_sigma_6, const float input_sigma_6_min) {
  int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
  float lambda_ = 1.0 - lambda;
  if (atom_i < atom_numbers) {
    ATOM_GROUP nl_i = nl[atom_i];
    int N = nl_i.atom_numbers;
    int atom_j;
    int int_x;
    int int_y;
    int int_z;
    UINT_VECTOR_LJ_FEP_TYPE r1 = uint_crd[atom_i], r2;
    VECTOR dr;
    float dr2, dr4, dr6;
    float dr_sc_A6, dr_sc_B6;
    float dr_sc_A, dr_sc_B;
    float dr_sc_A12, dr_sc_B12;
    float dr_2;
    float dr_4;
    float dr_8;
    float dr_6;
    float frc_abs = 0.;
    float AAij, ABij, BAij, BBij;
    float sigma_Aij, sigma_Bij;
    VECTOR frc_lin;
    VECTOR frc_record = {0., 0., 0.};

    float charge_i = r1.charge; // r1.charge;
    float charge_j;
    float dr_abs;
    float dr_1;
    float beta_dr;
    float beta_dr_sc_A, beta_dr_sc_B;
    float frc_cf_abs;

    float ene_lin = 0.0;
    float ene_lin2 = 0.0;

    int xA, yA, xB, yB;
    int atom_pair_LJ_type_A, atom_pair_LJ_type_B;

    int mask_i = r1.mask, mask_j;
    bool soft_core;
    for (int j = threadIdx.y; j < N; j = j + blockDim.y) {
      atom_j = nl_i.atom_serial[j];
      r2 = uint_crd[atom_j];
      charge_j = r2.charge;
      mask_j = r2.mask;

      int_x = r2.uint_x - r1.uint_x;
      int_y = r2.uint_y - r1.uint_y;
      int_z = r2.uint_z - r1.uint_z;
      dr.x = boxlength.x * int_x;
      dr.y = boxlength.y * int_y;
      dr.z = boxlength.z * int_z;
      dr_abs = norm3df(dr.x, dr.y, dr.z);
      if (dr_abs < cutoff) {
        yA = (r2.LJ_type_A - r1.LJ_type_A);
        xA = yA >> 31;
        yA = (yA ^ xA) - xA;
        xA = r2.LJ_type_A + r1.LJ_type_A;
        r2.LJ_type_A = (xA + yA) >> 1;
        xA = (xA - yA) >> 1;
        atom_pair_LJ_type_A = (r2.LJ_type_A * (r2.LJ_type_A + 1) >> 1) + xA;
        AAij = LJ_type_AA[atom_pair_LJ_type_A];
        ABij = LJ_type_AB[atom_pair_LJ_type_A];

        yB = (r2.LJ_type_B - r1.LJ_type_B);
        xB = yB >> 31;
        yB = (yB ^ xB) - xB;
        xB = r2.LJ_type_B + r1.LJ_type_B;
        r2.LJ_type_B = (xB + yB) >> 1;
        xB = (xB - yB) >> 1;
        atom_pair_LJ_type_B = (r2.LJ_type_B * (r2.LJ_type_B + 1) >> 1) + xB;
        BAij = LJ_type_BA[atom_pair_LJ_type_B];
        BBij = LJ_type_BB[atom_pair_LJ_type_B];

        soft_core = (mask_i != mask_j) || (BAij > 1e-6 && AAij < 1e-6) ||
                    (BAij < 1e-6 && AAij > 1e-6);
        if (!soft_core) {
          dr_1 = 1. / dr_abs;
          dr_2 = dr_1 * dr_1;
          dr_4 = dr_2 * dr_2;
          dr_8 = dr_4 * dr_4;
          dr_6 = dr_4 * dr_2;
          frc_abs = (-(lambda_ * AAij + lambda * BAij) * dr_6 +
                     (lambda_ * ABij + lambda * BBij)) *
                    dr_8;

          beta_dr = pme_beta * dr_abs;
          frc_cf_abs =
              beta_dr * sqrt_pi * expf(-beta_dr * beta_dr) + erfcf(beta_dr);
          frc_cf_abs = frc_cf_abs * dr_2 * dr_1;
          frc_cf_abs = charge_i * charge_j * frc_cf_abs;

          frc_abs = frc_abs - frc_cf_abs;

          ene_lin2 = ene_lin2 + charge_i * charge_j * erfcf(beta_dr) * dr_1;
          ene_lin =
              ene_lin + (0.083333333 * (lambda_ * AAij + lambda * BAij) * dr_6 -
                         0.166666666 * (lambda_ * ABij + lambda * BBij)) *
                            dr_6;
        } else {
          dr2 = dr_abs * dr_abs;
          dr4 = dr2 * dr2;
          dr6 = dr4 * dr2;
          if (AAij < 1e-6 || ABij < 1e-6)
            sigma_Aij = input_sigma_6;
          else
            sigma_Aij = max(0.5 * AAij / ABij, input_sigma_6_min);
          if (BAij < 1e-6 || BBij < 1e-6)
            sigma_Bij = input_sigma_6;
          else
            sigma_Bij = max(0.5 * BAij / BBij, input_sigma_6_min);

          dr_sc_A6 = 1.0 / (dr6 + alpha_lambda_p * sigma_Aij);
          dr_sc_B6 = 1.0 / (dr6 + alpha_lambda_p_ * sigma_Bij);
          dr_sc_A12 = dr_sc_A6 * dr_sc_A6;
          dr_sc_B12 = dr_sc_B6 * dr_sc_B6;

          frc_abs = dr4 * (lambda_ * (-AAij * dr_sc_A6 + ABij) * dr_sc_A12 +
                           lambda * (-BAij * dr_sc_B6 + BBij) * dr_sc_B12);

          dr_sc_A = powf(dr_sc_A6, 1.0 / 6.0);
          dr_sc_B = powf(dr_sc_B6, 1.0 / 6.0);
          beta_dr_sc_A = pme_beta / dr_sc_A;
          beta_dr_sc_B = pme_beta / dr_sc_B;

          frc_cf_abs =
              dr4 *
              (lambda_ *
                   (expf(-beta_dr_sc_A * beta_dr_sc_A) * sqrt_pi * pme_beta +
                    erfcf(beta_dr_sc_A) * dr_sc_A) *
                   dr_sc_A6 +
               lambda *
                   (expf(-beta_dr_sc_B * beta_dr_sc_B) * sqrt_pi * pme_beta +
                    erfcf(beta_dr_sc_B) * dr_sc_B) *
                   dr_sc_B6);
          frc_cf_abs = frc_cf_abs * charge_i * charge_j;

          frc_abs = frc_abs - frc_cf_abs;

          ene_lin2 = ene_lin2 + charge_i * charge_j *
                                    (lambda_ * erfcf(beta_dr_sc_A) * dr_sc_A +
                                     lambda * erfcf(beta_dr_sc_B) * dr_sc_B);

          ene_lin =
              ene_lin +
              lambda_ * (0.083333333 * AAij * dr_sc_A6 - 0.166666666 * ABij) *
                  dr_sc_A6 +
              lambda * (0.083333333 * BAij * dr_sc_B6 - 0.166666666 * BBij) *
                  dr_sc_B6;
        }

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

    atomicAdd(&atom_energy[atom_i], ene_lin + ene_lin2);
  }
}

void __global__
LJ_Soft_Core_Direct_CF_Force_With_LJ_Virial_Direct_CF_Energy_CUDA(
    const int atom_numbers, const ATOM_GROUP *nl,
    const UINT_VECTOR_LJ_FEP_TYPE *uint_crd, const VECTOR boxlength,
    const float *LJ_type_AA, const float *LJ_type_AB, const float *LJ_type_BA,
    const float *LJ_type_BB, const float cutoff, VECTOR *frc,
    const float pme_beta, const float sqrt_pi, float *atom_lj_virial,
    float *atom_direct_cf_energy, const float lambda,
    const float alpha_lambda_p, const float alpha_lambda_p_,
    const float input_sigma_6, const float input_sigma_6_min) {
  int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
  float lambda_ = 1.0 - lambda;
  if (atom_i < atom_numbers) {
    ATOM_GROUP nl_i = nl[atom_i];
    int N = nl_i.atom_numbers;
    int atom_j;
    int int_x;
    int int_y;
    int int_z;
    UINT_VECTOR_LJ_FEP_TYPE r1 = uint_crd[atom_i], r2;
    VECTOR dr;
    float dr2, dr4, dr6;
    float dr_sc_A6, dr_sc_B6;
    float dr_sc_A, dr_sc_B;
    float dr_sc_A12, dr_sc_B12;
    float dr_2;
    float dr_4;
    float dr_8;
    float dr_6;
    float frc_abs = 0.;
    float AAij, ABij, BAij, BBij;
    float sigma_Aij, sigma_Bij;
    VECTOR frc_lin;
    VECTOR frc_record = {0., 0., 0.};

    // CF
    float charge_i = r1.charge; // r1.charge;
    float charge_j;
    float dr_abs;
    float dr_1;
    float beta_dr;
    float beta_dr_sc_A, beta_dr_sc_B;
    float frc_cf_abs;

    float virial_lin = 0.0;
    float energy_lin = 0.0;

    int xA, yA, xB, yB;
    int atom_pair_LJ_type_A, atom_pair_LJ_type_B;
    bool soft_core;

    int mask_i = r1.mask, mask_j;
    for (int j = threadIdx.y; j < N; j = j + blockDim.y) {
      atom_j = nl_i.atom_serial[j];
      r2 = uint_crd[atom_j];
      // CF
      charge_j = r2.charge;
      mask_j = r2.mask;

      int_x = r2.uint_x - r1.uint_x;
      int_y = r2.uint_y - r1.uint_y;
      int_z = r2.uint_z - r1.uint_z;
      dr.x = boxlength.x * int_x;
      dr.y = boxlength.y * int_y;
      dr.z = boxlength.z * int_z;
      dr_abs = norm3df(dr.x, dr.y, dr.z);
      if (dr_abs < cutoff) {
        yA = (r2.LJ_type_A - r1.LJ_type_A);
        xA = yA >> 31;
        yA = (yA ^ xA) - xA;
        xA = r2.LJ_type_A + r1.LJ_type_A;
        r2.LJ_type_A = (xA + yA) >> 1;
        xA = (xA - yA) >> 1;
        atom_pair_LJ_type_A = (r2.LJ_type_A * (r2.LJ_type_A + 1) >> 1) + xA;
        AAij = LJ_type_AA[atom_pair_LJ_type_A];
        ABij = LJ_type_AB[atom_pair_LJ_type_A];

        yB = (r2.LJ_type_B - r1.LJ_type_B);
        xB = yB >> 31;
        yB = (yB ^ xB) - xB;
        xB = r2.LJ_type_B + r1.LJ_type_B;
        r2.LJ_type_B = (xB + yB) >> 1;
        xB = (xB - yB) >> 1;
        atom_pair_LJ_type_B = (r2.LJ_type_B * (r2.LJ_type_B + 1) >> 1) + xB;
        BAij = LJ_type_BA[atom_pair_LJ_type_B];
        BBij = LJ_type_BB[atom_pair_LJ_type_B];

        soft_core = (mask_i != mask_j) || (BAij > 1e-6 && AAij < 1e-6) ||
                    (BAij < 1e-6 && AAij > 1e-6);
        if (!soft_core) {
          dr_1 = 1. / dr_abs;
          dr_2 = dr_1 * dr_1;
          dr_4 = dr_2 * dr_2;
          dr_8 = dr_4 * dr_4;
          dr_6 = dr_4 * dr_2;
          frc_abs = (-(lambda_ * AAij + lambda * BAij) * dr_6 +
                     (lambda_ * ABij + lambda * BBij)) *
                    dr_8;

          beta_dr = pme_beta * dr_abs;
          frc_cf_abs =
              beta_dr * sqrt_pi * expf(-beta_dr * beta_dr) + erfcf(beta_dr);
          frc_cf_abs = frc_cf_abs * dr_2 * dr_1;
          frc_cf_abs = charge_i * charge_j * frc_cf_abs;

          virial_lin = virial_lin - frc_abs * dr_abs * dr_abs;

          frc_abs = frc_abs - frc_cf_abs;

          energy_lin = energy_lin + charge_i * charge_j * erfcf(beta_dr) * dr_1;
        } else {
          dr2 = dr_abs * dr_abs;
          dr4 = dr2 * dr2;
          dr6 = dr4 * dr2;
          if (AAij < 1e-6 || ABij < 1e-6)
            sigma_Aij = input_sigma_6;
          else
            sigma_Aij = max(0.5 * AAij / ABij, input_sigma_6_min);
          if (BAij < 1e-6 || BBij < 1e-6)
            sigma_Bij = input_sigma_6;
          else
            sigma_Bij = max(0.5 * BAij / BBij, input_sigma_6_min);

          dr_sc_A6 = 1.0 / (dr6 + alpha_lambda_p * sigma_Aij);
          dr_sc_B6 = 1.0 / (dr6 + alpha_lambda_p_ * sigma_Bij);
          dr_sc_A12 = dr_sc_A6 * dr_sc_A6;
          dr_sc_B12 = dr_sc_B6 * dr_sc_B6;

          frc_abs = dr4 * (lambda_ * (-AAij * dr_sc_A6 + ABij) * dr_sc_A12 +
                           lambda * (-BAij * dr_sc_B6 + BBij) * dr_sc_B12);

          dr_sc_A = powf(dr_sc_A6, 1.0 / 6.0);
          dr_sc_B = powf(dr_sc_B6, 1.0 / 6.0);
          beta_dr_sc_A = pme_beta / dr_sc_A;
          beta_dr_sc_B = pme_beta / dr_sc_B;

          frc_cf_abs =
              dr4 *
              (lambda_ *
                   (expf(-beta_dr_sc_A * beta_dr_sc_A) * sqrt_pi * pme_beta +
                    erfcf(beta_dr_sc_A) * dr_sc_A) *
                   dr_sc_A6 +
               lambda *
                   (expf(-beta_dr_sc_B * beta_dr_sc_B) * sqrt_pi * pme_beta +
                    erfcf(beta_dr_sc_B) * dr_sc_B) *
                   dr_sc_B6);
          frc_cf_abs = frc_cf_abs * charge_i * charge_j;

          virial_lin = virial_lin - frc_abs * dr_abs * dr_abs;

          frc_abs = frc_abs - frc_cf_abs;

          energy_lin =
              energy_lin + charge_i * charge_j *
                               (lambda_ * erfcf(beta_dr_sc_A) * dr_sc_A +
                                lambda * erfcf(beta_dr_sc_B) * dr_sc_B);
        }

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

    atomicAdd(&atom_direct_cf_energy[atom_i], energy_lin);
    atomicAdd(&atom_lj_virial[atom_i], virial_lin);
  }
}

static __global__ void
LJ_Soft_Core_Direct_CF_Force_With_Atom_Energy_And_LJ_Virial_Direct_CF_Energy_CUDA(
    const int atom_numbers, const ATOM_GROUP *nl,
    const UINT_VECTOR_LJ_FEP_TYPE *uint_crd, const VECTOR boxlength,
    const float *LJ_type_AA, const float *LJ_type_AB, const float *LJ_type_BA,
    const float *LJ_type_BB, const float cutoff, VECTOR *frc,
    const float pme_beta, const float sqrt_pi, float *atom_energy,
    float *atom_lj_virial, float *atom_direct_cf_energy, const float lambda,
    const float alpha_lambda_p, const float alpha_lambda_p_,
    const float input_sigma_6, const float input_sigma_6_min) {
  int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
  float lambda_ = 1.0 - lambda;
  if (atom_i < atom_numbers) {
    ATOM_GROUP nl_i = nl[atom_i];
    int N = nl_i.atom_numbers;
    int atom_j;
    int int_x;
    int int_y;
    int int_z;
    UINT_VECTOR_LJ_FEP_TYPE r1 = uint_crd[atom_i], r2;
    VECTOR dr;
    float dr2, dr4, dr6;
    float dr_sc_A6, dr_sc_B6;
    float dr_sc_A, dr_sc_B;
    float dr_sc_A12, dr_sc_B12;
    float dr_2;
    float dr_4;
    float dr_8;
    float dr_6;
    float frc_abs = 0.;
    float AAij, ABij, BAij, BBij;
    float sigma_Aij, sigma_Bij;
    VECTOR frc_lin;
    VECTOR frc_record = {0., 0., 0.};

    // CF
    float charge_i = r1.charge; // r1.charge;
    float charge_j;
    float dr_abs;
    float dr_1;
    float beta_dr;
    float beta_dr_sc_A, beta_dr_sc_B;
    float frc_cf_abs;
    //

    //能量
    float ene_lin = 0.0;
    float ene_lin2 = 0.0;
    float virial_lin = 0.0;

    int xA, yA, xB, yB;
    int atom_pair_LJ_type_A, atom_pair_LJ_type_B;
    int mask_i = r1.mask, mask_j;
    bool soft_core;
    for (int j = threadIdx.y; j < N; j = j + blockDim.y) {
      atom_j = nl_i.atom_serial[j];
      r2 = uint_crd[atom_j];
      charge_j = r2.charge;
      mask_j = r2.mask;

      int_x = r2.uint_x - r1.uint_x;
      int_y = r2.uint_y - r1.uint_y;
      int_z = r2.uint_z - r1.uint_z;
      dr.x = boxlength.x * int_x;
      dr.y = boxlength.y * int_y;
      dr.z = boxlength.z * int_z;
      dr_abs = norm3df(dr.x, dr.y, dr.z);
      if (dr_abs < cutoff) {
        yA = (r2.LJ_type_A - r1.LJ_type_A);
        xA = yA >> 31;
        yA = (yA ^ xA) - xA;
        xA = r2.LJ_type_A + r1.LJ_type_A;
        r2.LJ_type_A = (xA + yA) >> 1;
        xA = (xA - yA) >> 1;
        atom_pair_LJ_type_A = (r2.LJ_type_A * (r2.LJ_type_A + 1) >> 1) + xA;
        AAij = LJ_type_AA[atom_pair_LJ_type_A];
        ABij = LJ_type_AB[atom_pair_LJ_type_A];

        yB = (r2.LJ_type_B - r1.LJ_type_B);
        xB = yB >> 31;
        yB = (yB ^ xB) - xB;
        xB = r2.LJ_type_B + r1.LJ_type_B;
        r2.LJ_type_B = (xB + yB) >> 1;
        xB = (xB - yB) >> 1;
        atom_pair_LJ_type_B = (r2.LJ_type_B * (r2.LJ_type_B + 1) >> 1) + xB;
        BAij = LJ_type_BA[atom_pair_LJ_type_B];
        BBij = LJ_type_BB[atom_pair_LJ_type_B];

        soft_core = (mask_i != mask_j) || (BAij > 1e-6 && AAij < 1e-6) ||
                    (BAij < 1e-6 && AAij > 1e-6);
        if (!soft_core) {
          dr_1 = 1. / dr_abs;
          dr_2 = dr_1 * dr_1;
          dr_4 = dr_2 * dr_2;
          dr_8 = dr_4 * dr_4;
          dr_6 = dr_4 * dr_2;
          frc_abs = (-(lambda_ * AAij + lambda * BAij) * dr_6 +
                     (lambda_ * ABij + lambda * BBij)) *
                    dr_8;

          beta_dr = pme_beta * dr_abs;
          frc_cf_abs =
              beta_dr * sqrt_pi * expf(-beta_dr * beta_dr) + erfcf(beta_dr);
          frc_cf_abs = frc_cf_abs * dr_2 * dr_1;
          frc_cf_abs = charge_i * charge_j * frc_cf_abs;

          virial_lin = virial_lin - frc_abs * dr_abs * dr_abs;

          frc_abs = frc_abs - frc_cf_abs;

          ene_lin2 = ene_lin2 + charge_i * charge_j * erfcf(beta_dr) * dr_1;
          ene_lin =
              ene_lin + (0.083333333 * (lambda_ * AAij + lambda * BAij) * dr_6 -
                         0.166666666 * (lambda_ * ABij + lambda * BBij)) *
                            dr_6;
        } else {
          dr2 = dr_abs * dr_abs;
          dr4 = dr2 * dr2;
          dr6 = dr4 * dr2;
          if (AAij < 1e-6 || ABij < 1e-6)
            sigma_Aij = input_sigma_6;
          else
            sigma_Aij = max(0.5 * AAij / ABij, input_sigma_6_min);
          if (BAij < 1e-6 || BBij < 1e-6)
            sigma_Bij = input_sigma_6;
          else
            sigma_Bij = max(0.5 * BAij / BBij, input_sigma_6_min);

          dr_sc_A6 = 1.0 / (dr6 + alpha_lambda_p * sigma_Aij);
          dr_sc_B6 = 1.0 / (dr6 + alpha_lambda_p_ * sigma_Bij);
          dr_sc_A12 = dr_sc_A6 * dr_sc_A6;
          dr_sc_B12 = dr_sc_B6 * dr_sc_B6;

          frc_abs = dr4 * (lambda_ * (-AAij * dr_sc_A6 + ABij) * dr_sc_A12 +
                           lambda * (-BAij * dr_sc_B6 + BBij) * dr_sc_B12);

          dr_sc_A = powf(dr_sc_A6, 1.0 / 6.0);
          dr_sc_B = powf(dr_sc_B6, 1.0 / 6.0);
          beta_dr_sc_A = pme_beta / dr_sc_A;
          beta_dr_sc_B = pme_beta / dr_sc_B;

          frc_cf_abs =
              dr4 *
              (lambda_ *
                   (expf(-beta_dr_sc_A * beta_dr_sc_A) * sqrt_pi * pme_beta +
                    erfcf(beta_dr_sc_A) * dr_sc_A) *
                   dr_sc_A6 +
               lambda *
                   (expf(-beta_dr_sc_B * beta_dr_sc_B) * sqrt_pi * pme_beta +
                    erfcf(beta_dr_sc_B) * dr_sc_B) *
                   dr_sc_B6);
          frc_cf_abs = frc_cf_abs * charge_i * charge_j;

          virial_lin = virial_lin - frc_abs * dr_abs * dr_abs;

          frc_abs = frc_abs - frc_cf_abs;

          ene_lin2 = ene_lin2 + charge_i * charge_j *
                                    (lambda_ * erfcf(beta_dr_sc_A) * dr_sc_A +
                                     lambda * erfcf(beta_dr_sc_B) * dr_sc_B);

          ene_lin =
              ene_lin +
              lambda_ * (0.083333333 * AAij * dr_sc_A6 - 0.166666666 * ABij) *
                  dr_sc_A6 +
              lambda * (0.083333333 * BAij * dr_sc_B6 - 0.166666666 * BBij) *
                  dr_sc_B6;
        }

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

    atomicAdd(&atom_energy[atom_i], ene_lin + ene_lin2);
    atomicAdd(&atom_direct_cf_energy[atom_i], ene_lin2);
    atomicAdd(&atom_lj_virial[atom_i], virial_lin);
  }
}

static __global__ void LJ_Soft_Core_Energy_CUDA(
    const int atom_numbers, const ATOM_GROUP *nl,
    const UINT_VECTOR_LJ_FEP_TYPE *uint_crd, const VECTOR boxlength,
    const float *LJ_type_AA, const float *LJ_type_AB, const float *LJ_type_BA,
    const float *LJ_type_BB, const float cutoff, const float pme_beta,
    float *lj_ene, float *direct_ene, const float lambda,
    const float alpha_lambda_p, const float alpha_lambda_p_,
    const float input_sigma_6, const float input_sigma_6_min) {
  int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
  float lambda_ = 1.0 - lambda;
  if (atom_i < atom_numbers) {
    ATOM_GROUP nl_i = nl[atom_i];
    int N = nl_i.atom_numbers;
    int atom_j;
    int int_x;
    int int_y;
    int int_z;
    UINT_VECTOR_LJ_FEP_TYPE r1 = uint_crd[atom_i], r2;
    VECTOR dr;
    float dr2, dr4, dr6;
    float dr_sc_A6, dr_sc_B6;
    float dr_sc_A, dr_sc_B;
    float dr_2;
    float dr_4;
    float dr_6;
    float AAij, ABij, BAij, BBij;
    float sigma_Aij, sigma_Bij;

    float charge_i = r1.charge; // r1.charge;
    float charge_j;
    float dr_abs;
    float dr_1;
    float beta_dr;
    float beta_dr_sc_A, beta_dr_sc_B;

    float ene_lin = 0.0;
    float ene_lin2 = 0.0;

    int xA, yA, xB, yB;
    int atom_pair_LJ_type_A, atom_pair_LJ_type_B;
    int mask_i = r1.mask, mask_j;
    bool soft_core;
    for (int j = threadIdx.y; j < N; j = j + blockDim.y) {
      atom_j = nl_i.atom_serial[j];
      r2 = uint_crd[atom_j];
      charge_j = r2.charge;
      mask_j = r2.mask;

      int_x = r2.uint_x - r1.uint_x;
      int_y = r2.uint_y - r1.uint_y;
      int_z = r2.uint_z - r1.uint_z;
      dr.x = boxlength.x * int_x;
      dr.y = boxlength.y * int_y;
      dr.z = boxlength.z * int_z;
      dr_abs = norm3df(dr.x, dr.y, dr.z);
      if (dr_abs < cutoff) {
        yA = (r2.LJ_type_A - r1.LJ_type_A);
        xA = yA >> 31;
        yA = (yA ^ xA) - xA;
        xA = r2.LJ_type_A + r1.LJ_type_A;
        r2.LJ_type_A = (xA + yA) >> 1;
        xA = (xA - yA) >> 1;
        atom_pair_LJ_type_A = (r2.LJ_type_A * (r2.LJ_type_A + 1) >> 1) + xA;
        AAij = LJ_type_AA[atom_pair_LJ_type_A];
        ABij = LJ_type_AB[atom_pair_LJ_type_A];

        yB = (r2.LJ_type_B - r1.LJ_type_B);
        xB = yB >> 31;
        yB = (yB ^ xB) - xB;
        xB = r2.LJ_type_B + r1.LJ_type_B;
        r2.LJ_type_B = (xB + yB) >> 1;
        xB = (xB - yB) >> 1;
        atom_pair_LJ_type_B = (r2.LJ_type_B * (r2.LJ_type_B + 1) >> 1) + xB;
        BAij = LJ_type_BA[atom_pair_LJ_type_B];
        BBij = LJ_type_BB[atom_pair_LJ_type_B];

        soft_core = (mask_i != mask_j) || (BAij > 1e-6 && AAij < 1e-6) ||
                    (BAij < 1e-6 && AAij > 1e-6);
        if (!soft_core) {
          dr_1 = 1. / dr_abs;
          dr_2 = dr_1 * dr_1;
          dr_4 = dr_2 * dr_2;
          dr_6 = dr_4 * dr_2;

          beta_dr = pme_beta * dr_abs;

          ene_lin2 = ene_lin2 + charge_i * charge_j * erfcf(beta_dr) * dr_1;
          ene_lin =
              ene_lin + (0.083333333 * (lambda_ * AAij + lambda * BAij) * dr_6 -
                         0.166666666 * (lambda_ * ABij + lambda * BBij)) *
                            dr_6;
        } else {
          dr2 = dr_abs * dr_abs;
          dr4 = dr2 * dr2;
          dr6 = dr4 * dr2;
          if (AAij < 1e-6 || ABij < 1e-6)
            sigma_Aij = input_sigma_6;
          else
            sigma_Aij = max(0.5 * AAij / ABij, input_sigma_6_min);
          if (BAij < 1e-6 || BBij < 1e-6)
            sigma_Bij = input_sigma_6;
          else
            sigma_Bij = max(0.5 * BAij / BBij, input_sigma_6_min);

          dr_sc_A6 = 1.0 / (dr6 + alpha_lambda_p * sigma_Aij);
          dr_sc_B6 = 1.0 / (dr6 + alpha_lambda_p_ * sigma_Bij);

          dr_sc_A = powf(dr_sc_A6, 1.0 / 6.0);
          dr_sc_B = powf(dr_sc_B6, 1.0 / 6.0);
          beta_dr_sc_A = pme_beta / dr_sc_A;
          beta_dr_sc_B = pme_beta / dr_sc_B;

          ene_lin2 = ene_lin2 + charge_i * charge_j *
                                    (lambda_ * erfcf(beta_dr_sc_A) * dr_sc_A +
                                     lambda * erfcf(beta_dr_sc_B) * dr_sc_B);

          ene_lin =
              ene_lin +
              lambda_ * (0.083333333 * AAij * dr_sc_A6 - 0.166666666 * ABij) *
                  dr_sc_A6 +
              lambda * (0.083333333 * BAij * dr_sc_B6 - 0.166666666 * BBij) *
                  dr_sc_B6;
        }
      }
    } // atom_j cycle

    atomicAdd(&lj_ene[atom_i], ene_lin);
    atomicAdd(direct_ene, ene_lin2);
  }
}

static __global__ void LJ_Soft_Core_Energy_With_Coulumb_Direct_CUDA(
    const int atom_numbers, const ATOM_GROUP *nl,
    const UINT_VECTOR_LJ_FEP_TYPE *uint_crd, const VECTOR boxlength,
    const float *LJ_type_AA, const float *LJ_type_AB, const float *LJ_type_BA,
    const float *LJ_type_BB, const float cutoff, const float pme_beta,
    float *lj_ene_intersys, float *lj_ene_intrasys, float *direct_ene_intersys,
    float *direct_ene_intrasys, const float lambda, const float alpha_lambda_p,
    const float alpha_lambda_p_, const float input_sigma_6,
    const float input_sigma_6_min) {
  int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
  float lambda_ = 1.0 - lambda;
  if (atom_i < atom_numbers) {
    ATOM_GROUP nl_i = nl[atom_i];
    int N = nl_i.atom_numbers;
    int atom_j;
    UINT_VECTOR_LJ_FEP_TYPE r1 = uint_crd[atom_i], r2;
    VECTOR dr;
    float dr2, dr4, dr6;
    float dr_scA6, dr_scB6;
    float dr_sc_A6, dr_sc_B6;
    float dr_scA, dr_scB;
    float dr_sc_A, dr_sc_B;
    float vdw_A, vdw_B, ee_A, ee_B;
    float dr_2;
    float dr_4;
    float dr_6;
    float AAij, ABij, BAij, BBij;
    float sigma_Aij, sigma_Bij;

    float charge_i = r1.charge; // r1.charge;
    float charge_j;
    float dr_abs;
    float dr_1;
    float beta_dr;
    float beta_dr_sc_A, beta_dr_sc_B;

    float ene_lin_intersys = 0.0;
    float ene_lin2_intersys = 0.0;
    float ene_lin_intrasys = 0.0;
    float ene_lin2_intrasys = 0.0;

    float ene_tmp = 0.0;
    float ene_tmp_2 = 0.0;

    int xA, yA, xB, yB;
    int atom_pair_LJ_type_A, atom_pair_LJ_type_B;
    int mask_i = r1.mask, mask_j;
    bool soft_core;
    for (int j = threadIdx.y; j < N; j = j + blockDim.y) {
      atom_j = nl_i.atom_serial[j];
      r2 = uint_crd[atom_j];
      charge_j = r2.charge;
      mask_j = r2.mask;

      dr = Get_Periodic_Displacement(r1, r2, boxlength);

      dr_abs = norm3df(dr.x, dr.y, dr.z);
      // dr_abs = sqrt(dr.x * dr.x + dr.y * dr.y + dr.z * dr.z);
      if (dr_abs < cutoff) {
        yA = (r2.LJ_type_A - r1.LJ_type_A);
        xA = yA >> 31;
        yA = (yA ^ xA) - xA;
        xA = r2.LJ_type_A + r1.LJ_type_A;
        r2.LJ_type_A = (xA + yA) >> 1;
        xA = (xA - yA) >> 1;
        atom_pair_LJ_type_A = (r2.LJ_type_A * (r2.LJ_type_A + 1) >> 1) + xA;
        AAij = LJ_type_AA[atom_pair_LJ_type_A];
        ABij = LJ_type_AB[atom_pair_LJ_type_A];

        yB = (r2.LJ_type_B - r1.LJ_type_B);
        xB = yB >> 31;
        yB = (yB ^ xB) - xB;
        xB = r2.LJ_type_B + r1.LJ_type_B;
        r2.LJ_type_B = (xB + yB) >> 1;
        xB = (xB - yB) >> 1;
        atom_pair_LJ_type_B = (r2.LJ_type_B * (r2.LJ_type_B + 1) >> 1) + xB;
        BAij = LJ_type_BA[atom_pair_LJ_type_B];
        BBij = LJ_type_BB[atom_pair_LJ_type_B];

        soft_core = (mask_i != mask_j) || (BAij > 1e-6 && AAij < 1e-6) ||
                    (BAij < 1e-6 && AAij > 1e-6);

        if (!soft_core) {
          dr_1 = 1. / dr_abs;
          dr_2 = dr_1 * dr_1;
          dr_4 = dr_2 * dr_2;
          dr_6 = dr_4 * dr_2;

          beta_dr = pme_beta * dr_abs;

          ene_tmp_2 = charge_i * charge_j * erfcf(beta_dr) * dr_1;
          ene_tmp = (0.083333333 * (lambda_ * AAij + lambda * BAij) * dr_6 -
                     0.166666666 * (lambda_ * ABij + lambda * BBij)) *
                    dr_6;
        } else {
          dr2 = dr_abs * dr_abs;
          dr4 = dr2 * dr2;
          dr6 = dr4 * dr2;
          if (AAij < 1e-6 || ABij < 1e-6)
            sigma_Aij = input_sigma_6;
          else
            sigma_Aij = max(0.5 * AAij / ABij, input_sigma_6_min);
          if (BAij < 1e-6 || BBij < 1e-6)
            sigma_Bij = input_sigma_6;
          else
            sigma_Bij = max(0.5 * BAij / BBij, input_sigma_6_min);

          // printf("%f %f\n", sigma_Aij, sigma_Bij);
          // printf("%f %f %f %f\n", AAij / 12.0, ABij / 6.0, BAij / 12.0f, BBij
          // / 6.0f);

          dr_scA6 = (dr6 + alpha_lambda_p * sigma_Aij);
          dr_scB6 = (dr6 + alpha_lambda_p_ * sigma_Bij);

          dr_scA = powf(dr_scA6, 1.0 / 6.0);
          dr_scB = powf(dr_scB6, 1.0 / 6.0);

          if (dr_scA < cutoff) {
            dr_sc_A6 = 1.0 / dr_scA6;
            vdw_A =
                (0.083333333 * AAij * dr_sc_A6 - 0.166666666 * ABij) * dr_sc_A6;
            dr_sc_A = 1.0 / dr_scA;
            beta_dr_sc_A = pme_beta * dr_scA;
            ee_A = erfcf(beta_dr_sc_A) * dr_sc_A;
          } else {
            vdw_A = ee_A = 0.0f;
          }

          if (dr_scB < cutoff) {
            dr_sc_B6 = 1.0 / dr_scB6;
            vdw_B =
                (0.083333333 * BAij * dr_sc_B6 - 0.166666666 * BBij) * dr_sc_B6;
            dr_sc_B = 1.0 / dr_scB;
            beta_dr_sc_B = pme_beta * dr_scB;
            ee_B = erfc(beta_dr_sc_B) * dr_sc_B;
          } else {
            vdw_B = ee_B = 0.0f;
          }

          ene_tmp = lambda_ * vdw_A + lambda * vdw_B;
          ene_tmp_2 = charge_i * charge_j * (lambda_ * ee_A + lambda * ee_B);
        }

        if (mask_i == mask_j) {
          ene_lin_intrasys += ene_tmp;
          ene_lin2_intrasys += ene_tmp_2;
        } else {
          ene_lin_intersys += ene_tmp;
          ene_lin2_intersys += ene_tmp_2;
        }
      }
    } // atom_j cycle
    atomicAdd(&lj_ene_intersys[atom_i], ene_lin_intersys);
    atomicAdd(&lj_ene_intrasys[atom_i], ene_lin_intrasys);
    atomicAdd(direct_ene_intersys, ene_lin2_intersys);
    atomicAdd(direct_ene_intrasys, ene_lin2_intrasys);
  }
}

static __global__ void LJ_Soft_Core_With_Drect_Columb_dH_dlambda_CUDA(
    const int atom_numbers, const ATOM_GROUP *nl,
    const UINT_VECTOR_LJ_FEP_TYPE *uint_crd, const VECTOR boxlength,
    const float *LJ_type_AA, const float *LJ_type_AB, const float *LJ_type_BA,
    const float *LJ_type_BB, const float *charge_B_A, const float cutoff,
    float *sigma_dH_dlambda_lj, float *sigma_dH_dlambda_direct,
    const float pme_beta, const float sqrt_pi, const float lambda,
    const float alpha_lambda_p, const float alpha_lambda_p_,
    const float alpha_lambda_p_1, const float alpha_lambda_p_1_,
    const float input_sigma_6, const float input_sigma_6_min, const float p) {
  int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
  float lambda_ = 1.0 - lambda;
  if (atom_i < atom_numbers) {
    ATOM_GROUP nl_i = nl[atom_i];
    int N = nl_i.atom_numbers;
    int atom_j;
    int int_x;
    int int_y;
    int int_z;
    UINT_VECTOR_LJ_FEP_TYPE r1 = uint_crd[atom_i], r2;
    VECTOR dr;
    float dr2, dr4, dr6;
    float dr_sc_A6, dr_sc_B6;
    float dr_sc_A, dr_sc_B;
    float dr_sc_A12, dr_sc_B12;
    float dr_2;
    float dr_4;
    float dr_6;
    float AAij, ABij, BAij, BBij;
    float sigma_Aij, sigma_Bij;
    int mask_i, mask_j;

    float charge_i = r1.charge; // r1.charge;
    float charge_j;
    float dr_abs;
    float beta_dr_sc_A, beta_dr_sc_B;

    float charge_B_A_i = charge_B_A[atom_i], charge_B_A_j;

    float dH_dlambda_lj_sum = 0.0;
    float dH_dlambda_cf_sum = 0.0;
    float dH_dlambda_lj = 0.0;
    float dH_dlambda_cf = 0.0;

    int xA, yA, xB, yB;
    int atom_pair_LJ_type_A, atom_pair_LJ_type_B;
    mask_i = r1.mask;
    bool soft_core;

    for (int j = threadIdx.y; j < N; j = j + blockDim.y) {
      atom_j = nl_i.atom_serial[j];
      r2 = uint_crd[atom_j];
      mask_j = r2.mask;
      charge_j = r2.charge;
      charge_B_A_j = charge_B_A[atom_j];

      int_x = r2.uint_x - r1.uint_x;
      int_y = r2.uint_y - r1.uint_y;
      int_z = r2.uint_z - r1.uint_z;
      dr.x = boxlength.x * int_x;
      dr.y = boxlength.y * int_y;
      dr.z = boxlength.z * int_z;
      dr_abs = norm3df(dr.x, dr.y, dr.z);
      if (dr_abs < cutoff) {
        yA = (r2.LJ_type_A - r1.LJ_type_A);
        xA = yA >> 31;
        yA = (yA ^ xA) - xA;
        xA = r2.LJ_type_A + r1.LJ_type_A;
        r2.LJ_type_A = (xA + yA) >> 1;
        xA = (xA - yA) >> 1;
        atom_pair_LJ_type_A = (r2.LJ_type_A * (r2.LJ_type_A + 1) >> 1) + xA;
        AAij = LJ_type_AA[atom_pair_LJ_type_A];
        ABij = LJ_type_AB[atom_pair_LJ_type_A];

        yB = (r2.LJ_type_B - r1.LJ_type_B);
        xB = yB >> 31;
        yB = (yB ^ xB) - xB;
        xB = r2.LJ_type_B + r1.LJ_type_B;
        r2.LJ_type_B = (xB + yB) >> 1;
        xB = (xB - yB) >> 1;
        atom_pair_LJ_type_B = (r2.LJ_type_B * (r2.LJ_type_B + 1) >> 1) + xB;
        BAij = LJ_type_BA[atom_pair_LJ_type_B];
        BBij = LJ_type_BB[atom_pair_LJ_type_B];
        dr2 = dr_abs * dr_abs;

        soft_core = (mask_i != mask_j) || (BAij > 1e-6 && AAij < 1e-6) ||
                    (BAij < 1e-6 && AAij > 1e-6);
        if (soft_core) {
          dr4 = dr2 * dr2;
          dr6 = dr4 * dr2;
          if (AAij < 1e-6 || ABij < 1e-6)
            sigma_Aij = input_sigma_6;
          else
            sigma_Aij = max(0.5 * AAij / ABij, input_sigma_6_min);
          if (BAij < 1e-6 || BBij < 1e-6)
            sigma_Bij = input_sigma_6;
          else
            sigma_Bij = max(0.5 * BAij / BBij, input_sigma_6_min);
          dr_sc_A6 = 1.0 / (dr6 + alpha_lambda_p * sigma_Aij);
          dr_sc_B6 = 1.0 / (dr6 + alpha_lambda_p_ * sigma_Bij);
          dr_sc_A12 = dr_sc_A6 * dr_sc_A6;
          dr_sc_B12 = dr_sc_B6 * dr_sc_B6;

          dH_dlambda_lj =
              dr_sc_B12 * 0.083333333 * BAij - 0.166666666 * BBij * dr_sc_B6 -
              0.083333333 * AAij * dr_sc_A12 + 0.166666666 * ABij * dr_sc_A6;
          dH_dlambda_lj =
              dH_dlambda_lj +
              lambda * dr_sc_B12 * p * sigma_Bij * alpha_lambda_p_1_ *
                  (0.166666666 * BAij * dr_sc_B6 - 0.166666666 * BBij) +
              lambda_ * dr_sc_A12 * p * sigma_Aij * alpha_lambda_p_1 *
                  (-0.166666666 * AAij * dr_sc_A6 + 0.166666666 * ABij);

          dH_dlambda_lj_sum += dH_dlambda_lj;

          dr_sc_A = powf(dr_sc_A6, 1.0 / 6.0);
          dr_sc_B = powf(dr_sc_B6, 1.0 / 6.0);
          beta_dr_sc_A = pme_beta / dr_sc_A;
          beta_dr_sc_B = pme_beta / dr_sc_B;

          dr_sc_A = powf(dr_sc_A6, 1.0 / 6.0);
          dr_sc_B = powf(dr_sc_B6, 1.0 / 6.0);
          beta_dr_sc_A = pme_beta / dr_sc_A;
          beta_dr_sc_B = pme_beta / dr_sc_B;

          dH_dlambda_cf =
              lambda * p * sigma_Bij * alpha_lambda_p_1_ * dr_sc_B6 *
              (pme_beta * sqrt_pi * expf(-beta_dr_sc_B * beta_dr_sc_B) +
               erfcf(beta_dr_sc_B) * 0.166666666 * dr_sc_B);

          dH_dlambda_cf =
              dH_dlambda_cf -
              lambda_ * p * sigma_Aij * alpha_lambda_p_1 * dr_sc_A6 *
                  (pme_beta * sqrt_pi * expf(-beta_dr_sc_A * beta_dr_sc_A) +
                   erfcf(beta_dr_sc_A) * 0.166666666 * dr_sc_A);

          dH_dlambda_cf = dH_dlambda_cf - erfcf(beta_dr_sc_A) * dr_sc_A +
                          erfcf(beta_dr_sc_B) * dr_sc_B;

          dH_dlambda_cf = dH_dlambda_cf * charge_i * charge_j;

          dH_dlambda_cf = dH_dlambda_cf +
                          (charge_B_A_j * charge_i + charge_j * charge_B_A_i) *
                              (lambda_ * erfcf(beta_dr_sc_A) * dr_sc_A +
                               lambda * erfcf(beta_dr_sc_B) * dr_sc_B);
          // printf("%f %f\n", record, dH_dlambda_cf - record);

          dH_dlambda_cf_sum += dH_dlambda_cf;
        } else {
          dr_2 = 1.0 / dr2;
          dr_4 = dr_2 * dr_2;
          dr_6 = dr_4 * dr_2;
          dH_dlambda_lj = (0.083333333 * (BAij - AAij) * dr_6 -
                           0.166666666 * (BBij - ABij)) *
                          dr_6;
          dH_dlambda_cf = erfcf(pme_beta * dr_abs) *
                          (charge_B_A_j * charge_i + charge_j * charge_B_A_i);

          dH_dlambda_lj_sum += dH_dlambda_lj;
          dH_dlambda_cf_sum += dH_dlambda_cf;
        }
      }
    } // atom_j cycle
    atomicAdd(sigma_dH_dlambda_lj, dH_dlambda_lj_sum);
    atomicAdd(sigma_dH_dlambda_direct, dH_dlambda_cf_sum);
  }
}

static __global__ void
LJ_Soft_Core_With_Drect_Columb_dH_dlambda_Charge_Unpertubated_CUDA(
    const int atom_numbers, const ATOM_GROUP *nl,
    const UINT_VECTOR_LJ_FEP_TYPE *uint_crd, const VECTOR boxlength,
    const float *LJ_type_AA, const float *LJ_type_AB, const float *LJ_type_BA,
    const float *LJ_type_BB, const float cutoff, float *sigma_dH_dlambda_lj,
    float *sigma_dH_dlambda_direct, const float pme_beta, const float sqrt_pi,
    const float lambda, const float alpha_lambda_p, const float alpha_lambda_p_,
    const float alpha_lambda_p_1, const float alpha_lambda_p_1_,
    const float input_sigma_6, const float input_sigma_6_min, const float p) {
  int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
  float lambda_ = 1.0 - lambda;
  if (atom_i < atom_numbers) {
    ATOM_GROUP nl_i = nl[atom_i];
    int N = nl_i.atom_numbers;
    int atom_j;
    int int_x;
    int int_y;
    int int_z;
    UINT_VECTOR_LJ_FEP_TYPE r1 = uint_crd[atom_i], r2;
    VECTOR dr;
    float dr2, dr4, dr6;
    float dr_sc_A6, dr_sc_B6;
    float dr_sc_A, dr_sc_B;
    float dr_sc_A12, dr_sc_B12;
    float dr_2;
    float dr_4;
    float dr_6;
    float AAij, ABij, BAij, BBij;
    float sigma_Aij, sigma_Bij;
    int mask_i = r1.mask, mask_j;

    float charge_i = r1.charge; // r1.charge;
    float charge_j;
    float dr_abs;
    float beta_dr_sc_A, beta_dr_sc_B;

    float dH_dlambda_lj_sum = 0.0;
    float dH_dlambda_cf_sum = 0.0;
    float dH_dlambda_lj = 0.0;
    float dH_dlambda_cf = 0.0;

    int xA, yA, xB, yB;
    int atom_pair_LJ_type_A, atom_pair_LJ_type_B;
    bool soft_core;

    for (int j = threadIdx.y; j < N; j = j + blockDim.y) {
      atom_j = nl_i.atom_serial[j];
      r2 = uint_crd[atom_j];
      mask_j = r2.mask;
      charge_j = r2.charge;

      int_x = r2.uint_x - r1.uint_x;
      int_y = r2.uint_y - r1.uint_y;
      int_z = r2.uint_z - r1.uint_z;
      dr.x = boxlength.x * int_x;
      dr.y = boxlength.y * int_y;
      dr.z = boxlength.z * int_z;
      dr_abs = norm3df(dr.x, dr.y, dr.z);
      if (dr_abs < cutoff) {
        yA = (r2.LJ_type_A - r1.LJ_type_A);
        xA = yA >> 31;
        yA = (yA ^ xA) - xA;
        xA = r2.LJ_type_A + r1.LJ_type_A;
        r2.LJ_type_A = (xA + yA) >> 1;
        xA = (xA - yA) >> 1;
        atom_pair_LJ_type_A = (r2.LJ_type_A * (r2.LJ_type_A + 1) >> 1) + xA;
        AAij = LJ_type_AA[atom_pair_LJ_type_A];
        ABij = LJ_type_AB[atom_pair_LJ_type_A];

        yB = (r2.LJ_type_B - r1.LJ_type_B);
        xB = yB >> 31;
        yB = (yB ^ xB) - xB;
        xB = r2.LJ_type_B + r1.LJ_type_B;
        r2.LJ_type_B = (xB + yB) >> 1;
        xB = (xB - yB) >> 1;
        atom_pair_LJ_type_B = (r2.LJ_type_B * (r2.LJ_type_B + 1) >> 1) + xB;
        BAij = LJ_type_BA[atom_pair_LJ_type_B];
        BBij = LJ_type_BB[atom_pair_LJ_type_B];
        dr2 = dr_abs * dr_abs;

        soft_core = (mask_i != mask_j) || (BAij > 1e-6 && AAij < 1e-6) ||
                    (BAij < 1e-6 && AAij > 1e-6);
        if (soft_core) {
          dr4 = dr2 * dr2;
          dr6 = dr4 * dr2;
          if (AAij < 1e-6 || ABij < 1e-6)
            sigma_Aij = input_sigma_6;
          else
            sigma_Aij = max(0.5 * AAij / ABij, input_sigma_6_min);
          if (BAij < 1e-6 || BBij < 1e-6)
            sigma_Bij = input_sigma_6;
          else
            sigma_Bij = max(0.5 * BAij / BBij, input_sigma_6_min);
          dr_sc_A6 = 1.0 / (dr6 + alpha_lambda_p * sigma_Aij);
          dr_sc_B6 = 1.0 / (dr6 + alpha_lambda_p_ * sigma_Bij);
          dr_sc_A12 = dr_sc_A6 * dr_sc_A6;
          dr_sc_B12 = dr_sc_B6 * dr_sc_B6;

          dH_dlambda_lj =
              dr_sc_B12 * 0.083333333 * BAij - 0.166666666 * BBij * dr_sc_B6 -
              0.083333333 * AAij * dr_sc_A12 + 0.166666666 * ABij * dr_sc_A6;
          dH_dlambda_lj =
              dH_dlambda_lj +
              lambda * dr_sc_B12 * p * sigma_Bij * alpha_lambda_p_1_ *
                  (0.166666666 * BAij * dr_sc_B6 - 0.166666666 * BBij) +
              lambda_ * dr_sc_A12 * p * sigma_Aij * alpha_lambda_p_1 *
                  (-0.166666666 * AAij * dr_sc_A6 + 0.166666666 * ABij);

          dH_dlambda_lj_sum += dH_dlambda_lj;

          dr_sc_A = powf(dr_sc_A6, 1.0 / 6.0);
          dr_sc_B = powf(dr_sc_B6, 1.0 / 6.0);
          beta_dr_sc_A = pme_beta / dr_sc_A;
          beta_dr_sc_B = pme_beta / dr_sc_B;

          dH_dlambda_cf =
              lambda * p * sigma_Bij * alpha_lambda_p_1_ * dr_sc_B6 *
              (pme_beta * sqrt_pi * expf(-beta_dr_sc_B * beta_dr_sc_B) +
               erfcf(beta_dr_sc_B) * 0.166666666 * dr_sc_B);

          dH_dlambda_cf =
              dH_dlambda_cf -
              lambda_ * p * sigma_Aij * alpha_lambda_p_1 * dr_sc_A6 *
                  (pme_beta * sqrt_pi * expf(-beta_dr_sc_A * beta_dr_sc_A) +
                   erfcf(beta_dr_sc_A) * 0.166666666 * dr_sc_A);

          dH_dlambda_cf = dH_dlambda_cf - erfcf(beta_dr_sc_A) * dr_sc_A +
                          erfcf(beta_dr_sc_B) * dr_sc_B;

          dH_dlambda_cf_sum =
              dH_dlambda_cf_sum + dH_dlambda_cf * charge_i * charge_j;
        } else {
          dr_2 = 1.0 / dr2;
          dr_4 = dr_2 * dr_2;
          dr_6 = dr_4 * dr_2;
          dH_dlambda_lj = (0.083333333 * (BAij - AAij) * dr_6 -
                           0.166666666 * (BBij - ABij)) *
                          dr_6;

          dH_dlambda_lj_sum += dH_dlambda_lj;
        }
      }
    } // atom_j cycle
    atomicAdd(sigma_dH_dlambda_lj, dH_dlambda_lj_sum);
    atomicAdd(sigma_dH_dlambda_direct, dH_dlambda_cf_sum);
  }
}

void LJ_SOFT_CORE::Initial(CONTROLLER *controller, float cutoff,
                           VECTOR box_length, char *module_name) {
  if (module_name == NULL) {
    strcpy(this->module_name, "LJ_soft_core");
  } else {
    strcpy(this->module_name, module_name);
  }
  controller[0].printf(
      "START INITIALIZING FEP SOFT CORE FOR LJ AND COULOMB:\n");
  if (controller[0].Command_Exist(this->module_name, "in_file")) {
    if (controller[0].Command_Exist("lambda_lj")) {
      this->lambda = atof(controller[0].Command("lambda_lj"));
      controller->printf("    FEP lj lambda: %f\n", this->lambda);
    } else {
      controller->printf("    Error: FEP lambda of LJ must be given for the "
                         "calculation of SOFT CORE.\n");
      getchar();
      fcloseall();
      exit(0);
    }

    if (controller[0].Command_Exist("soft_core_alpha")) {
      this->alpha = atof(controller[0].Command("soft_core_alpha"));
      controller->printf("    FEP soft core alpha: %f\n", this->alpha);
    } else {
      controller->printf(
          "    FEP soft core alpha is set to default value 0.5\n");
      this->alpha = 0.5;
    }

    if (controller[0].Command_Exist("soft_core_powfer")) {
      this->p = atof(controller[0].Command("soft_core_powfer"));
      controller->printf("    FEP soft core powfer: %f\n", this->p);
    } else {
      controller->printf(
          "    FEP soft core powfer is set to default value 1.0.\n");
      this->p = 1.0;
    }

    if (controller[0].Command_Exist("soft_core_sigma")) {
      this->sigma = atof(controller[0].Command("soft_core_sigma"));
      controller->printf("    FEP soft core sigma: %f\n", this->sigma);
    } else {
      controller->printf(
          "    FEP soft core sigma is set to default value 3.0\n");
      this->sigma = 3.0;
    }
    if (controller[0].Command_Exist("soft_core_sigma_min")) {
      this->sigma_min = atof(controller[0].Command("soft_core_sigma_min"));
      controller->printf("    FEP soft core sigma min: %f\n", this->sigma_min);
    } else {
      controller->printf(
          "    FEP soft core sigma min is set to default value 0.0\n");
      this->sigma_min = 0.0;
    }

    FILE *fp = NULL;
    Open_File_Safely(&fp, controller[0].Command(this->module_name, "in_file"),
                     "r");

    int toscan = fscanf(fp, "%d %d %d", &atom_numbers, &atom_type_numbers_A,
                        &atom_type_numbers_B);
    controller[0].printf("    atom_numbers is %d\n", atom_numbers);
    controller[0].printf(
        "    atom_LJ_type_number_A is %d, atom_LJ_type_number_B is %d\n",
        atom_type_numbers_A, atom_type_numbers_B);
    pair_type_numbers_A = atom_type_numbers_A * (atom_type_numbers_A + 1) / 2;
    pair_type_numbers_B = atom_type_numbers_B * (atom_type_numbers_B + 1) / 2;
    this->thread_LJ = {8, 32};
    LJ_Soft_Core_Malloc();

    for (int i = 0; i < pair_type_numbers_A; i++) {
      toscan = fscanf(fp, "%f", h_LJ_AA + i);
      h_LJ_AA[i] *= 12.0f;
    }
    for (int i = 0; i < pair_type_numbers_A; i++) {
      toscan = fscanf(fp, "%f", h_LJ_AB + i);
      h_LJ_AB[i] *= 6.0f;
    }
    for (int i = 0; i < pair_type_numbers_B; ++i) {
      toscan = fscanf(fp, "%f", h_LJ_BA + i);
      h_LJ_BA[i] *= 12.0f;
    }
    for (int i = 0; i < pair_type_numbers_B; ++i) {
      toscan = fscanf(fp, "%f", h_LJ_BB + i);
      h_LJ_BB[i] *= 6.0f;
    }
    for (int i = 0; i < atom_numbers; i++) {
      toscan = fscanf(fp, "%d %d", h_atom_LJ_type_A + i, h_atom_LJ_type_B + i);
    }
    fclose(fp);

    if (controller[0].Command_Exist("subsys_division_in_file")) {
      FILE *fp = NULL;
      controller->printf("     Start reading subsystem division "
                         "information:\n");
      Open_File_Safely(&fp, controller[0].Command("subsys_division_in_file"),
                       "r");
      int atom_numbers = 0;
      char lin[CHAR_LENGTH_MAX];
      char *get_ret = fgets(lin, CHAR_LENGTH_MAX, fp);
      toscan = sscanf(lin, "%d", &atom_numbers);
      if (this->atom_numbers > 0 && this->atom_numbers != atom_numbers) {
        controller->printf("        Error: atom_numbers is not equal: %d %d\n",
                           this->atom_numbers, atom_numbers);
        getchar();
        exit(1);
      } else if (this->atom_numbers == 0) {
        this->atom_numbers = atom_numbers;
      }
      for (int i = 0; i < atom_numbers; i++) {
        toscan = fscanf(fp, "%d", &h_subsys_division[i]);
      }
      controller->printf("    End reading subsystem division information\n\n");
      fclose(fp);
    } else {
      controller[0].printf("    subsystem mask is set to 0 as default\n");
      for (int i = 0; i < atom_numbers; i++) {
        h_subsys_division[i] = 0;
      }
    }

    Parameter_Host_To_Device();
    is_initialized = 1;
    alpha_lambda_p = alpha * powf(lambda, p);
    alpha_lambda_p_ = alpha * powf(1 - lambda, p);
    sigma_6 = powf(sigma, 6);
    sigma_6_min = powf(sigma_min, 6);
    alpha_lambda_p_1 = alpha * powf(lambda, p - 1);
    alpha_lambda_p_1_ = alpha * powf(1.0 - lambda, p - 1);

    pme_tolerance = 0.00001;
    if (controller[0].Command_Exist("PME_Direct_Tolerance"))
      pme_tolerance = atof(controller[0].Command("PME_Direct_Tolerance"));

    pme_beta = Get_Beta_LJ(cutoff, pme_tolerance);
  }
  if (is_initialized) {
    this->cutoff = cutoff;
    this->uint_dr_to_dr_cof = 1.0f / CONSTANT_UINT_MAX_FLOAT * box_length;
    Cuda_Malloc_Safely((void **)&uint_crd_with_LJ,
                       sizeof(UINT_VECTOR_LJ_FEP_TYPE) * atom_numbers);
    Copy_LJ_Type_And_Mask_To_New_Crd<<<ceilf((float)this->atom_numbers / 32),
                                       32>>>(atom_numbers, uint_crd_with_LJ,
                                             d_atom_LJ_type_A, d_atom_LJ_type_B,
                                             d_subsys_division);

    controller[0].printf("    Start initializing long range LJ correction\n");
    long_range_factor = 0;
    float *d_factor = NULL;
    Cuda_Malloc_Safely((void **)&d_factor, sizeof(float));
    Reset_List(d_factor, 0.0f, 1, 1);
    Total_C6_Get<<<{4, 4}, {32, 32}>>>(atom_numbers, d_atom_LJ_type_A,
                                       d_atom_LJ_type_B, d_LJ_AB, d_LJ_BB,
                                       d_factor, this->lambda);
    cudaMemcpy(&long_range_factor, d_factor, sizeof(float),
               cudaMemcpyDeviceToHost);
    Reset_List(d_factor, 0.0f, 1, 1);
    Total_C6_B_A_Get<<<{4, 4}, {32, 32}>>>(atom_numbers, d_atom_LJ_type_A,
                                           d_atom_LJ_type_B, d_LJ_AB, d_LJ_BB,
                                           d_factor);
    cudaMemcpy(&long_range_factor_TI, d_factor, sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaFree(d_factor);

    long_range_factor *=
        -2.0f / 3.0f * CONSTANT_Pi / cutoff / cutoff / cutoff / 6.0f;
    long_range_factor_TI *=
        -2.0f / 3.0f * CONSTANT_Pi / cutoff / cutoff / cutoff / 6.0f;
    this->volume = box_length.x * box_length.y * box_length.z;
    controller[0].printf("        long range correction factor is: %e\n",
                         long_range_factor);
    controller[0].printf("    End initializing long range LJ correction\n");
  }
  if (is_initialized && !is_controller_printf_initialized) {
    controller[0].Step_Print_Initial("LJ(sc.)", "%.2f");
    controller[0].Step_Print_Initial("LR_corr(sc.)", "%.2f");
    is_controller_printf_initialized = 1;
    controller[0].printf("    structure last modify date is %d\n",
                         last_modify_date);
  }
  controller[0].printf("END INITIALIZING LENNADR JONES INFORMATION\n\n");
}

void LJ_SOFT_CORE::LJ_Soft_Core_Malloc() {
  Malloc_Safely((void **)&h_LJ_energy_atom, sizeof(float) * atom_numbers);
  Malloc_Safely((void **)&h_atom_LJ_type_A, sizeof(int) * atom_numbers);
  Malloc_Safely((void **)&h_atom_LJ_type_B, sizeof(int) * atom_numbers);
  Malloc_Safely((void **)&h_LJ_AA, sizeof(float) * pair_type_numbers_A);
  Malloc_Safely((void **)&h_LJ_AB, sizeof(float) * pair_type_numbers_A);
  Malloc_Safely((void **)&h_LJ_BA, sizeof(float) * pair_type_numbers_B);
  Malloc_Safely((void **)&h_LJ_BB, sizeof(float) * pair_type_numbers_B);
  Malloc_Safely((void **)&h_subsys_division, sizeof(int) * atom_numbers);

  Cuda_Malloc_Safely((void **)&d_LJ_energy_sum, sizeof(float));
  Cuda_Malloc_Safely((void **)&d_LJ_energy_atom, sizeof(float) * atom_numbers);
  Cuda_Malloc_Safely((void **)&d_atom_LJ_type_A, sizeof(int) * atom_numbers);
  Cuda_Malloc_Safely((void **)&d_atom_LJ_type_B, sizeof(int) * atom_numbers);
  Cuda_Malloc_Safely((void **)&d_LJ_AA, sizeof(float) * pair_type_numbers_A);
  Cuda_Malloc_Safely((void **)&d_LJ_AB, sizeof(float) * pair_type_numbers_A);
  Cuda_Malloc_Safely((void **)&d_LJ_BA, sizeof(float) * pair_type_numbers_B);
  Cuda_Malloc_Safely((void **)&d_LJ_BB, sizeof(float) * pair_type_numbers_B);
  Cuda_Malloc_Safely((void **)&d_subsys_division, sizeof(int) * atom_numbers);

  Malloc_Safely((void **)&h_LJ_energy_atom_intersys,
                sizeof(float) * atom_numbers);
  Malloc_Safely((void **)&h_LJ_energy_atom_intrasys,
                sizeof(float) * atom_numbers);

  Cuda_Malloc_Safely((void **)&d_direct_ene_sum_intersys, sizeof(float));
  Cuda_Malloc_Safely((void **)&d_direct_ene_sum_intrasys, sizeof(float));
  Cuda_Malloc_Safely((void **)&d_LJ_energy_sum_intersys, sizeof(float));
  Cuda_Malloc_Safely((void **)&d_LJ_energy_sum_intrasys, sizeof(float));
  Cuda_Malloc_Safely((void **)&d_LJ_energy_atom_intersys,
                     sizeof(float) * atom_numbers);
  Cuda_Malloc_Safely((void **)&d_LJ_energy_atom_intrasys,
                     sizeof(float) * atom_numbers);

  Malloc_Safely((void **)&h_sigma_of_dH_dlambda_lj, sizeof(float));
  Malloc_Safely((void **)&h_sigma_of_dH_dlambda_direct, sizeof(float));

  Cuda_Malloc_Safely((void **)&d_sigma_of_dH_dlambda_lj, sizeof(float));
  Cuda_Malloc_Safely((void **)&d_sigma_of_dH_dlambda_direct, sizeof(float));
  Cuda_Malloc_Safely((void **)&d_long_range_correction, sizeof(float));
}

void LJ_SOFT_CORE::Parameter_Host_To_Device() {
  cudaMemcpy(d_LJ_AB, h_LJ_AB, sizeof(float) * pair_type_numbers_A,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_LJ_AA, h_LJ_AA, sizeof(float) * pair_type_numbers_A,
             cudaMemcpyHostToDevice);

  cudaMemcpy(d_LJ_BA, h_LJ_BA, sizeof(float) * pair_type_numbers_B,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_LJ_BB, h_LJ_BB, sizeof(float) * pair_type_numbers_B,
             cudaMemcpyHostToDevice);

  cudaMemcpy(d_atom_LJ_type_A, h_atom_LJ_type_A, sizeof(int) * atom_numbers,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_atom_LJ_type_B, h_atom_LJ_type_B, sizeof(int) * atom_numbers,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_subsys_division, h_subsys_division, sizeof(int) * atom_numbers,
             cudaMemcpyHostToDevice);
}

void LJ_SOFT_CORE::Clear() {
  if (is_initialized) {
    is_initialized = 0;

    free(h_atom_LJ_type_A);
    free(h_atom_LJ_type_B);
    cudaFree(d_atom_LJ_type_A);
    cudaFree(d_atom_LJ_type_B);

    free(h_LJ_AA);
    free(h_LJ_AB);
    free(h_LJ_BA);
    free(h_LJ_BB);
    cudaFree(d_LJ_AA);
    cudaFree(d_LJ_AB);
    cudaFree(d_LJ_BA);
    cudaFree(d_LJ_BB);

    free(h_LJ_energy_atom);
    cudaFree(d_LJ_energy_atom);
    cudaFree(d_LJ_energy_sum);

    cudaFree(uint_crd_with_LJ);

    free(h_subsys_division);
    cudaFree(d_subsys_division);

    h_atom_LJ_type_A = NULL;
    d_atom_LJ_type_A = NULL;
    h_atom_LJ_type_B = NULL;
    d_atom_LJ_type_B = NULL;

    h_LJ_AA = NULL;
    h_LJ_AB = NULL;
    d_LJ_AA = NULL;
    d_LJ_AB = NULL;

    h_LJ_BA = NULL;
    h_LJ_BB = NULL;
    d_LJ_BA = NULL;
    d_LJ_BB = NULL;

    h_LJ_energy_atom = NULL;
    d_LJ_energy_atom = NULL;
    d_LJ_energy_sum = NULL;

    uint_crd_with_LJ = NULL;
    h_subsys_division = NULL;
    d_subsys_division = NULL;
  }
}

void LJ_SOFT_CORE::LJ_Soft_Core_Force_With_PME_Direct_Force(
    const int atom_numbers, const UINT_VECTOR_LJ_FEP_TYPE *uint_crd,
    const VECTOR scaler, VECTOR *frc, const ATOM_GROUP *nl, const float cutoff,
    const float pme_beta) {
  if (is_initialized)
    LJ_Soft_Core_Force_With_Direct_CF_CUDA<<<
        (unsigned int)ceilf((float)atom_numbers / thread_LJ.x), thread_LJ>>>(
        atom_numbers, nl, uint_crd, scaler, d_LJ_AA, d_LJ_AB, d_LJ_BA, d_LJ_BB,
        cutoff, frc, pme_beta, TWO_DIVIDED_BY_SQRT_PI, lambda, alpha_lambda_p,
        alpha_lambda_p_, sigma_6, sigma_6_min);
}

void LJ_SOFT_CORE::LJ_Soft_Core_PME_Direct_Force_With_Atom_Energy(
    const int atom_numbers, const UINT_VECTOR_LJ_FEP_TYPE *uint_crd,
    const VECTOR scaler, VECTOR *frc, const ATOM_GROUP *nl, const float cutoff,
    const float pme_beta, float *atom_energy) {
  if (is_initialized)
    LJ_Soft_Core_Direct_CF_Force_With_Atom_Energy_CUDA<<<
        (unsigned int)ceilf((float)atom_numbers / thread_LJ.x), thread_LJ>>>(
        atom_numbers, nl, uint_crd, scaler, d_LJ_AA, d_LJ_AB, d_LJ_BA, d_LJ_BB,
        cutoff, frc, pme_beta, TWO_DIVIDED_BY_SQRT_PI, atom_energy, lambda,
        alpha_lambda_p, alpha_lambda_p_, sigma_6, sigma_6_min);
}

void LJ_SOFT_CORE::LJ_Soft_Core_PME_Direct_Force_With_Atom_Energy_And_Virial(
    const int atom_numbers, const UNSIGNED_INT_VECTOR *uint_crd,
    const float *charge, VECTOR *frc, const ATOM_GROUP *nl,
    const float pme_beta, const int need_atom_energy, float *atom_energy,
    const int need_virial, float *atom_lj_virial,
    float *atom_direct_pme_energy) {
  if (is_initialized) {
    Copy_Crd_And_Charge_To_New_Crd<<<
        (unsigned int)ceilf((float)atom_numbers / 32), 32>>>(
        atom_numbers, uint_crd, uint_crd_with_LJ, charge);
    if (!need_atom_energy > 0 && !need_virial > 0) {
      LJ_Soft_Core_Force_With_PME_Direct_Force(atom_numbers, uint_crd_with_LJ,
                                               uint_dr_to_dr_cof, frc, nl,
                                               cutoff, pme_beta);
    } else if (need_atom_energy > 0 && !need_virial > 0) {
      LJ_Soft_Core_PME_Direct_Force_With_Atom_Energy(
          atom_numbers, uint_crd_with_LJ, uint_dr_to_dr_cof, frc, nl, cutoff,
          pme_beta, atom_energy);
    } else if (!need_atom_energy > 0 && need_virial > 0) {
      Reset_List(atom_direct_pme_energy, 0.0f, atom_numbers, 1024);
      LJ_Soft_Core_Direct_CF_Force_With_LJ_Virial_Direct_CF_Energy_CUDA<<<
          (unsigned int)ceilf((float)atom_numbers / thread_LJ.x), thread_LJ>>>(
          atom_numbers, nl, uint_crd_with_LJ, uint_dr_to_dr_cof, d_LJ_AA,
          d_LJ_AB, d_LJ_BA, d_LJ_BB, cutoff, frc, pme_beta,
          TWO_DIVIDED_BY_SQRT_PI, atom_lj_virial, atom_direct_pme_energy,
          lambda, alpha_lambda_p, alpha_lambda_p_, sigma_6, sigma_6_min);
    } else {
      Reset_List(atom_direct_pme_energy, 0.0f, atom_numbers, 1024);
      LJ_Soft_Core_Direct_CF_Force_With_Atom_Energy_And_LJ_Virial_Direct_CF_Energy_CUDA<<<
          (unsigned int)ceilf((float)atom_numbers / thread_LJ.x), thread_LJ>>>(
          atom_numbers, nl, uint_crd_with_LJ, uint_dr_to_dr_cof, d_LJ_AA,
          d_LJ_AB, d_LJ_BA, d_LJ_BB, cutoff, frc, pme_beta,
          TWO_DIVIDED_BY_SQRT_PI, atom_energy, atom_lj_virial,
          atom_direct_pme_energy, lambda, alpha_lambda_p, alpha_lambda_p_,
          sigma_6, sigma_6_min);
    }
  }
}

float LJ_SOFT_CORE::Get_Energy(const UNSIGNED_INT_VECTOR *uint_crd,
                               const ATOM_GROUP *nl, const float pme_beta,
                               const float *charge, float *direct_ene_sum,
                               int is_download) {
  if (is_initialized) {
    // printf("This func! is_download = %d\n", is_download);
    // printf("thread_LJ.x: %u %u\n", thread_LJ.x, thread_LJ.y);
    Copy_Crd_And_Charge_To_New_Crd<<<
        (unsigned int)ceilf((float)atom_numbers / 32), 32>>>(
        atom_numbers, uint_crd, uint_crd_with_LJ, charge);

    Reset_List(d_LJ_energy_atom, 0., atom_numbers, 1024);
    LJ_Soft_Core_Energy_CUDA<<<
        (unsigned int)ceilf((float)atom_numbers / thread_LJ.x), thread_LJ>>>(
        atom_numbers, nl, uint_crd_with_LJ, uint_dr_to_dr_cof, d_LJ_AA, d_LJ_AB,
        d_LJ_BA, d_LJ_BB, cutoff, pme_beta, d_LJ_energy_atom, direct_ene_sum,
        lambda, alpha_lambda_p, alpha_lambda_p_, sigma_6, sigma_6_min);

    Sum_Of_List(d_LJ_energy_atom, d_LJ_energy_sum, atom_numbers);

    long_range_correction = long_range_factor / this->volume;

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

float LJ_SOFT_CORE::Get_Energy_With_Coulomb_Direct(
    const UNSIGNED_INT_VECTOR *uint_crd, const ATOM_GROUP *nl,
    const float *charge, int is_download) {
  if (is_initialized) {
    Copy_Crd_And_Charge_To_New_Crd<<<
        (unsigned int)ceilf((float)atom_numbers / 32), 32>>>(
        atom_numbers, uint_crd, uint_crd_with_LJ, charge);

    Reset_List(d_LJ_energy_atom_intersys, 0., atom_numbers, 1024);
    Reset_List(d_LJ_energy_atom_intrasys, 0., atom_numbers, 1024);
    cudaMemset(d_direct_ene_sum_intersys, 0, sizeof(float));
    cudaMemset(d_direct_ene_sum_intrasys, 0, sizeof(float));

    LJ_Soft_Core_Energy_With_Coulumb_Direct_CUDA<<<
        (unsigned int)ceilf((float)atom_numbers / thread_LJ.x), thread_LJ>>>(
        atom_numbers, nl, uint_crd_with_LJ, uint_dr_to_dr_cof, d_LJ_AA, d_LJ_AB,
        d_LJ_BA, d_LJ_BB, cutoff, pme_beta, d_LJ_energy_atom_intersys,
        d_LJ_energy_atom_intrasys, d_direct_ene_sum_intersys,
        d_direct_ene_sum_intrasys, lambda, alpha_lambda_p, alpha_lambda_p_,
        sigma_6, sigma_6_min);
    // cudaDeviceSynchronize();
    // getchar();

    Sum_Of_List(d_LJ_energy_atom_intersys, d_LJ_energy_sum_intersys,
                atom_numbers);
    Sum_Of_List(d_LJ_energy_atom_intrasys, d_LJ_energy_sum_intrasys,
                atom_numbers);

    if (is_download) {
      cudaMemcpy(&h_LJ_energy_sum_intersys, this->d_LJ_energy_sum_intersys,
                 sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(&h_LJ_energy_sum_intrasys, this->d_LJ_energy_sum_intrasys,
                 sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(&h_direct_ene_sum_intersys, this->d_direct_ene_sum_intersys,
                 sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(&h_direct_ene_sum_intrasys, this->d_direct_ene_sum_intrasys,
                 sizeof(float), cudaMemcpyDeviceToHost);

      h_LJ_energy_sum = h_LJ_energy_sum_intersys + h_LJ_energy_sum_intrasys;
      return h_LJ_energy_sum;
    } else {
      return 0.0;
    }
  } else {
    return NAN;
  }
}

float LJ_SOFT_CORE::Get_Partial_H_Partial_Lambda_With_Columb_Direct(
    const UNSIGNED_INT_VECTOR *uint_crd, const float *charge,
    const ATOM_GROUP *nl, const float *charge_B_A, const int charge_pertubated,
    int is_download) {
  if (is_initialized) {
    Copy_Crd_And_Charge_To_New_Crd<<<
        (unsigned int)ceilf((float)atom_numbers / 32), 32>>>(
        atom_numbers, uint_crd, uint_crd_with_LJ, charge);

    cudaMemset(d_sigma_of_dH_dlambda_lj, 0, sizeof(float));
    cudaMemset(d_sigma_of_dH_dlambda_direct, 0, sizeof(float));
    if (charge_pertubated > 0) {
      LJ_Soft_Core_With_Drect_Columb_dH_dlambda_CUDA<<<
          (unsigned int)ceilf((float)atom_numbers / thread_LJ.x), thread_LJ>>>(
          atom_numbers, nl, uint_crd_with_LJ, uint_dr_to_dr_cof, d_LJ_AA,
          d_LJ_AB, d_LJ_BA, d_LJ_BB, charge_B_A, cutoff,
          d_sigma_of_dH_dlambda_lj, d_sigma_of_dH_dlambda_direct, pme_beta,
          ONE_DIVIDED_BY_3_SQRT_PI, lambda, alpha_lambda_p, alpha_lambda_p_,
          alpha_lambda_p_1, alpha_lambda_p_1_, sigma_6, sigma_6_min, p);
    } else {
      LJ_Soft_Core_With_Drect_Columb_dH_dlambda_Charge_Unpertubated_CUDA<<<
          (unsigned int)ceilf((float)atom_numbers / thread_LJ.x), thread_LJ>>>(
          atom_numbers, nl, uint_crd_with_LJ, uint_dr_to_dr_cof, d_LJ_AA,
          d_LJ_AB, d_LJ_BA, d_LJ_BB, cutoff, d_sigma_of_dH_dlambda_lj,
          d_sigma_of_dH_dlambda_direct, pme_beta, ONE_DIVIDED_BY_3_SQRT_PI,
          lambda, alpha_lambda_p, alpha_lambda_p_, alpha_lambda_p_1,
          alpha_lambda_p_1_, sigma_6, sigma_6_min, p);
    }

    if (is_download) {
      cudaMemcpy(h_sigma_of_dH_dlambda_lj, d_sigma_of_dH_dlambda_lj,
                 sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_sigma_of_dH_dlambda_direct, d_sigma_of_dH_dlambda_direct,
                 sizeof(float), cudaMemcpyDeviceToHost);
      return *h_sigma_of_dH_dlambda_lj;
    } else {
      return 0;
    }
  } else {
    return NAN;
  }
}

void LJ_SOFT_CORE::Update_Volume(VECTOR box_length) {
  if (!is_initialized)
    return;
  this->uint_dr_to_dr_cof = 1.0f / CONSTANT_UINT_MAX_FLOAT * box_length;
  this->volume = box_length.x * box_length.y * box_length.z;
}

void LJ_SOFT_CORE::Long_Range_Correction(float volume) {
  if (is_initialized) {
    device_add<<<1, 1>>>(d_LJ_energy_sum, long_range_factor / volume);
  }
}

void LJ_SOFT_CORE::Long_Range_Correction(int need_pressure, float *d_virial,
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

float LJ_SOFT_CORE::Long_Range_Correction() {
  if (is_initialized) {
    return long_range_factor / this->volume;
  } else {
    return NAN;
  }
}

float LJ_SOFT_CORE::Partial_H_Partial_Lambda_Long_Range_Correction() {
  if (is_initialized) {
    cudaMemset(d_long_range_correction, 0, sizeof(float));
    device_add<<<1, 1>>>(d_long_range_correction,
                         long_range_factor_TI / volume);
    cudaMemcpy(&long_range_correction, d_long_range_correction, sizeof(float),
               cudaMemcpyDeviceToHost);
    // printf("long range correction: %f\n", *long_range_correction);
    return long_range_correction;
  } else {
    return NAN;
  }
}
