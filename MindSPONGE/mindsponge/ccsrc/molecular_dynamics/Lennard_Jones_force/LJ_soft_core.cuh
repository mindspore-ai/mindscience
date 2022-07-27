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

#ifndef LJ_SOFT_CORE_CUH
#define LJ_SOFT_CORE_CUH
#include "../common.cuh"
#include "../control.cuh"

//用于计算LJ_Force时使用的坐标和记录的原子LJ种类序号与原子电荷
#ifndef UINT_VECTOR_LJ_FEP_TYPE_DEFINE
#define UINT_VECTOR_LJ_FEP_TYPE_DEFINE
struct UINT_VECTOR_LJ_FEP_TYPE {
  unsigned int uint_x;
  unsigned int uint_y;
  unsigned int uint_z;
  int LJ_type_A;
  int LJ_type_B;
  int mask;
  float charge;
};
__device__ __host__ VECTOR Get_Periodic_Displacement(
    const UINT_VECTOR_LJ_FEP_TYPE uvec_a, const UINT_VECTOR_LJ_FEP_TYPE uvec_b,
    const VECTOR scaler);
__global__ void Copy_LJ_Type_To_New_Crd(const int atom_numbers,
                                        UINT_VECTOR_LJ_FEP_TYPE *new_crd,
                                        const int *LJ_type);
__global__ void Copy_Crd_And_Charge_To_New_Crd(const int atom_numbers,
                                               const UNSIGNED_INT_VECTOR *crd,
                                               UINT_VECTOR_LJ_FEP_TYPE *new_crd,
                                               const float *charge);
__global__ void Copy_Crd_To_New_Crd(const int atom_numbers,
                                    const UNSIGNED_INT_VECTOR *crd,
                                    UINT_VECTOR_LJ_FEP_TYPE *new_crd);
#endif

struct LJ_SOFT_CORE {
  char module_name[CHAR_LENGTH_MAX];
  int is_initialized = 0;
  int is_controller_printf_initialized = 0;
  int last_modify_date = 20210730;

  int atom_numbers = 0;
  int atom_type_numbers_A = 0;
  int atom_type_numbers_B = 0;
  int pair_type_numbers_A = 0;
  int pair_type_numbers_B = 0;

  int *h_atom_LJ_type_A = NULL;
  int *h_atom_LJ_type_B = NULL;
  int *d_atom_LJ_type_A = NULL;
  int *d_atom_LJ_type_B = NULL;

  float *h_LJ_AA = NULL;
  float *h_LJ_AB = NULL;
  float *h_LJ_BA = NULL;
  float *h_LJ_BB = NULL;
  float *d_LJ_AA = NULL;
  float *d_LJ_AB = NULL;
  float *d_LJ_BA = NULL;
  float *d_LJ_BB = NULL;

  float *h_LJ_energy_atom = NULL;
  float h_LJ_energy_sum = 0;
  float *d_LJ_energy_atom = NULL;
  float *d_LJ_energy_sum = NULL;

  int *d_subsys_division;
  int *h_subsys_division;

  float *h_LJ_energy_atom_intersys = NULL;
  float *h_LJ_energy_atom_intrasys = NULL;
  float h_LJ_energy_sum_intersys = 0;
  float h_LJ_energy_sum_intrasys = 0;
  float *d_LJ_energy_atom_intersys = NULL;
  float *d_LJ_energy_atom_intrasys = NULL;
  float *d_LJ_energy_sum_intersys = NULL;
  float *d_LJ_energy_sum_intrasys = NULL;

  float *d_direct_ene_sum_intersys = NULL;
  float *d_direct_ene_sum_intrasys = NULL;
  float h_direct_ene_sum = 0.0;
  float h_direct_ene_sum_intersys = 0.0;
  float h_direct_ene_sum_intrasys = 0.0;

  float *d_long_range_correction;
  float long_range_correction = 0.0;

  float lambda;
  float alpha;
  float p;
  float alpha_lambda_p;
  float alpha_lambda_p_;
  float alpha_lambda_p_1;
  float alpha_lambda_p_1_;
  float sigma_6;
  float sigma;
  float sigma_min;
  float sigma_6_min;

  float pme_tolerance;
  float pme_beta;

  float *h_sigma_of_dH_dlambda_lj = NULL;
  float *d_sigma_of_dH_dlambda_lj = NULL;

  float *h_sigma_of_dH_dlambda_direct = NULL;
  float *d_sigma_of_dH_dlambda_direct = NULL;

  dim3 thread_LJ = {8u, 32u};

  float cutoff = 10.0;
  VECTOR uint_dr_to_dr_cof;
  float volume = 0;
  UINT_VECTOR_LJ_FEP_TYPE *uint_crd_with_LJ = NULL;
  float long_range_factor = 0.0;
  float long_range_factor_TI = 0.0;

  void Initial(CONTROLLER *controller, float cutoff, VECTOR box_length,
               char *module_name = NULL);

  void LJ_Soft_Core_Malloc();

  void Clear();

  void Parameter_Host_To_Device();

  void LJ_Soft_Core_Force_With_PME_Direct_Force(
      const int atom_numbers, const UINT_VECTOR_LJ_FEP_TYPE *uint_crd,
      const VECTOR scaler, VECTOR *frc, const ATOM_GROUP *nl,
      const float cutoff, const float pme_beta);

  void LJ_Soft_Core_PME_Direct_Force_With_Atom_Energy(
      const int atom_numbers, const UINT_VECTOR_LJ_FEP_TYPE *uint_crd,
      const VECTOR scaler, VECTOR *frc, const ATOM_GROUP *nl,
      const float cutoff, const float pme_beta, float *atom_energy);

  void LJ_Soft_Core_PME_Direct_Force_With_Atom_Energy_And_Virial(
      const int atom_numbers, const UNSIGNED_INT_VECTOR *uint_crd,
      const float *charge, VECTOR *frc, const ATOM_GROUP *nl,
      const float pme_beta, const int need_atom_energy, float *atom_energy,
      const int need_virial, float *atom_lj_virial,
      float *atom_direct_pme_energy);

  float Get_Energy(const UNSIGNED_INT_VECTOR *uint_crd, const ATOM_GROUP *nl,
                   const float pme_beta, const float *charge,
                   float *direct_ene_sum, int is_download = 1);

  float Get_Energy_With_Coulomb_Direct(const UNSIGNED_INT_VECTOR *uint_crd,
                                       const ATOM_GROUP *nl,
                                       const float *charge,
                                       int is_download = 1);

  void Update_Volume(VECTOR box_length);

  void Long_Range_Correction(int need_pressure, float *d_virial,
                             int need_potential, float *d_potential);

  void Long_Range_Correction(float volume);

  float Long_Range_Correction();

  float Get_Partial_H_Partial_Lambda_With_Columb_Direct(
      const UNSIGNED_INT_VECTOR *uint_crd, const float *charge,
      const ATOM_GROUP *nl, const float *charge_B_A,
      const int charge_pertubated = 0, int is_download = 1);

  float Partial_H_Partial_Lambda_Long_Range_Correction();
};
#endif
