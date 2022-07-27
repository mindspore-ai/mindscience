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

#ifndef BOND_SOFT_CUH
#define BOND_SOFT_CUH
#include "../common.cuh"
#include "../control.cuh"

struct BOND_SOFT {
  char module_name[CHAR_LENGTH_MAX];
  int is_initialized = 0;
  int is_controller_printf_initialized = 0;
  int last_modify_date = 20210730;

  int soft_bond_numbers = 0;
  int *h_atom_a = NULL;
  int *d_atom_a = NULL;
  int *h_atom_b = NULL;
  int *d_atom_b = NULL;
  int *h_ABmask = NULL;
  int *d_ABmask = NULL;
  float *d_k = NULL;
  float *h_k = NULL;
  float *d_r0 = NULL;
  float *h_r0 = NULL;
  float lambda;
  float alpha;

  float *h_soft_bond_ene = NULL;
  float *d_soft_bond_ene = NULL;
  float *d_sigma_of_soft_bond_ene = NULL;
  float *h_sigma_of_soft_bond_ene = NULL;

  float *h_soft_bond_dH_dlambda = NULL;
  float *d_soft_bond_dH_dlambda = NULL;
  float *h_sigma_of_dH_dlambda = NULL;
  float *d_sigma_of_dH_dlambda = NULL;

  int threads_per_block = 128;

  void Initial(CONTROLLER *controller, const char *module_name = NULL);

  void Clear();

  void Memory_Allocate();

  void Parameter_Host_To_Device();

  void Soft_Bond_Force_With_Atom_Energy_And_Virial(
      const UNSIGNED_INT_VECTOR *uint_crd, const VECTOR scaler, VECTOR *frc,
      float *atom_energy, float *atom_virial);

  float Get_Energy(const UNSIGNED_INT_VECTOR *unit_crd, const VECTOR scaler,
                   int is_download = 1);

  float Get_Partial_H_Partial_Lambda(const UNSIGNED_INT_VECTOR *uint_crd,
                                     const VECTOR scaler, int is_download = 1);
};

#endif
