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

#include "bond_soft.cuh"

static __global__ void Soft_Bond_Force_With_Atom_Energy_And_Virial_CUDA(
    const int bond_numbers, const UNSIGNED_INT_VECTOR *uint_crd,
    const VECTOR scaler, const int *atom_a, const int *atom_b,
    const float *bond_k, const float *bond_r0, const int *AB_mask, VECTOR *frc,
    float *atom_energy, float *atom_virial, const float lambda,
    const float alpha) {
  int bond_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (bond_i < bond_numbers) {
    int atom_i = atom_a[bond_i];
    int atom_j = atom_b[bond_i];

    float k = bond_k[bond_i];
    float r0 = bond_r0[bond_i];
    int ABmask = AB_mask[bond_i];
    float tmp_lambda = (ABmask == 0 ? 1.0 - lambda : lambda);
    float tmp_lambda_ = 1.0 - tmp_lambda;

    VECTOR dr =
        Get_Periodic_Displacement(uint_crd[atom_i], uint_crd[atom_j], scaler);
    float abs_r = norm3df(dr.x, dr.y, dr.z);
    float r_1 = 1. / abs_r;

    float temp_denominator =
        1 + alpha * tmp_lambda_ * (abs_r - r0) * (abs_r - r0);
    float tempf =
        2 * k * tmp_lambda * (abs_r - r0) / temp_denominator / temp_denominator;

    VECTOR f = tempf * r_1 * dr;

    atomicAdd(&frc[atom_i].x, -f.x);
    atomicAdd(&frc[atom_i].y, -f.y);
    atomicAdd(&frc[atom_i].z, -f.z);

    atomicAdd(&frc[atom_j].x, f.x);
    atomicAdd(&frc[atom_j].y, f.y);
    atomicAdd(&frc[atom_j].z, f.z);

    atomicAdd(&atom_virial[atom_i], -tempf * abs_r);
    atomicAdd(&atom_energy[atom_i],
              tmp_lambda * k * (abs_r - r0) * (abs_r - r0) / temp_denominator);
  }
}

static __global__ void
Soft_Bond_Energy_CUDA(const int bond_numbers,
                      const UNSIGNED_INT_VECTOR *uint_crd, const VECTOR scaler,
                      const int *atom_a, const int *atom_b, const float *bond_k,
                      const float *bond_r0, const int *AB_mask, float *bond_ene,
                      const float lambda, const float alpha) {
  int bond_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (bond_i < bond_numbers) {
    int atom_i = atom_a[bond_i];
    int atom_j = atom_b[bond_i];

    float k = bond_k[bond_i];
    float r0 = bond_r0[bond_i];
    int ABmask = AB_mask[bond_i];

    float tmp_lambda = (ABmask == 0 ? 1.0 - lambda : lambda);

    VECTOR dr =
        Get_Periodic_Displacement(uint_crd[atom_i], uint_crd[atom_j], scaler);

    float r1 = norm3df(dr.x, dr.y, dr.z);
    float tempf = r1 - r0;
    float tempf2 = tempf * tempf;

    bond_ene[bond_i] =
        tmp_lambda * k * tempf2 / (1 + alpha * (1 - tmp_lambda) * tempf2);
  }
}

static __global__ void Soft_Bond_dH_dlambda_CUDA(
    const int bond_numbers, const UNSIGNED_INT_VECTOR *uint_crd,
    const VECTOR scaler, const int *atom_a, const int *atom_b,
    const float *bond_k, const float *bond_r0, const int *AB_mask,
    float *dH_dlambda, const float lambda, const float alpha) {
  int bond_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (bond_i < bond_numbers) {
    int atom_i = atom_a[bond_i];
    int atom_j = atom_b[bond_i];

    float k = bond_k[bond_i];
    float r0 = bond_r0[bond_i];
    int ABmask = AB_mask[bond_i];

    VECTOR dr =
        Get_Periodic_Displacement(uint_crd[atom_i], uint_crd[atom_j], scaler);

    float r1 = norm3df(dr.x, dr.y, dr.z);
    float tmp_lambda = (ABmask == 0 ? 1.0 - lambda : lambda);
    float tmp_sign = (ABmask == 0 ? -1.0 : 1.0);
    float tmp_lambda_ = 1.0 - tmp_lambda;
    float tempf = r1 - r0;
    float tempf2 = tempf * tempf;
    float temp_denominator = 1.0 / (1 + alpha * tmp_lambda_ * tempf2);
    float dH_dlambda_abs =
        k * tempf2 * temp_denominator * temp_denominator * (1 + alpha * tempf2);

    dH_dlambda[bond_i] = dH_dlambda_abs * tmp_sign;
  }
}

void BOND_SOFT::Initial(CONTROLLER *controller, const char *module_name) {
  controller[0].printf("START INITIALIZING BOND SOFT:\n");
  if (module_name == NULL) {
    strcpy(this->module_name, "bond_soft");
  } else {
    strcpy(this->module_name, module_name);
  }
  if (controller[0].Command_Exist(this->module_name, "in_file")) {
    if (controller[0].Command_Exist("lambda_bond")) {
      this->lambda = atof(controller[0].Command("lambda_bond"));
    } else {
      printf("Error: FEP lambda of bond must be given for the calculation of "
             "SOFT BOND.\n");
      getchar();
    }
    if (controller[0].Command_Exist("soft_bond_alpha")) {
      this->alpha = atof(controller[0].Command("soft_bond_alpha"));
    } else {
      printf("Warning: FEP alpha of soft bond missing for the calculation of "
             "SOFT BOND, set to default value 0.0.\n");
      this->alpha = 0.0;
    }
    FILE *fp = NULL;
    Open_File_Safely(&fp, controller[0].Command(this->module_name, "in_file"),
                     "r");
    int toscan = fscanf(fp, "%d", &soft_bond_numbers);
    controller[0].printf("    soft_bond_numbers is %d\n", soft_bond_numbers);
    Memory_Allocate();
    for (int i = 0; i < soft_bond_numbers; i++) {
      toscan = fscanf(fp, "%d %d %f %f %d", h_atom_a + i, h_atom_b + i, h_k + i,
                      h_r0 + i, h_ABmask + i);
    }
    fclose(fp);
    Parameter_Host_To_Device();
    is_initialized = 1;
  }

  if (is_initialized && !is_controller_printf_initialized) {
    controller[0].Step_Print_Initial(this->module_name, "%.2f");
    is_controller_printf_initialized = 1;
    controller[0].printf("    structure last modify date is %d\n",
                         last_modify_date);
  }

  controller[0].printf("END INITIALIZING SOFT BOND\n\n");
}

void BOND_SOFT::Memory_Allocate() {
  if (!Malloc_Safely((void **)&(this->h_atom_a),
                     sizeof(int) * this->soft_bond_numbers))
    printf("        Error occurs when malloc BOND_SOFT::h_atom_a in "
           "BOND_SOFT::Memory_Allocate");
  if (!Malloc_Safely((void **)&(this->h_atom_b),
                     sizeof(int) * this->soft_bond_numbers))
    printf("        Error occurs when malloc BOND_SOFT::h_atom_b in "
           "BOND_SOFT::Memory_Allocate");
  if (!Malloc_Safely((void **)&(this->h_k),
                     sizeof(float) * this->soft_bond_numbers))
    printf("        Error occurs when malloc BOND_SOFT::h_k in "
           "BOND_SOFT::Memory_Allocate");
  if (!Malloc_Safely((void **)&(this->h_r0),
                     sizeof(float) * this->soft_bond_numbers))
    printf("        Error occurs when malloc BOND_SOFT::h_r0 in "
           "BOND_SOFT::Memory_Allocate");
  if (!Malloc_Safely((void **)&(this->h_ABmask),
                     sizeof(int) * this->soft_bond_numbers))
    printf("        Error occurs when malloc BOND_SOFT::h_ABmask in "
           "BOND_SOFT::Memory_Allocate");
  if (!Malloc_Safely((void **)&(this->h_soft_bond_ene),
                     sizeof(float) * this->soft_bond_numbers))
    printf("        Error occurs when malloc BOND_SOFT::h_soft_bond_ene in "
           "BOND_SOFT::Memory_Allocate");
  if (!Malloc_Safely((void **)&(this->h_sigma_of_soft_bond_ene), sizeof(float)))
    printf("        Error occurs when malloc "
           "BOND_SOFT::h_sigma_of_soft_bond_ene in BOND_SOFT::Memory_Allocate");
  if (!Malloc_Safely((void **)&(this->h_soft_bond_dH_dlambda),
                     sizeof(float) * this->soft_bond_numbers))
    printf("        Error occurs when malloc "
           "BOND_SOFT::h_soft_bond_dH_dlambda in BOND_SOFT::Memory_Allocate");
  if (!Malloc_Safely((void **)&(this->h_sigma_of_dH_dlambda), sizeof(float)))
    printf("       Error occurs when malloc "
           "BOND_SOFT::h_sigma_of_dH_dlambda in BOND_SOFT::Memory_Allocate");

  if (!Cuda_Malloc_Safely((void **)&this->d_atom_a,
                          sizeof(int) * this->soft_bond_numbers))
    printf("       Error occurs when CUDA malloc BOND_SOFT::d_atom_a in "
           "BOND_SOFT::Memory_Allocate");
  if (!Cuda_Malloc_Safely((void **)&this->d_atom_b,
                          sizeof(int) * this->soft_bond_numbers))
    printf("        Error occurs when CUDA malloc BOND_SOFT::d_atom_b in "
           "BOND_SOFT::Memory_Allocate");
  if (!Cuda_Malloc_Safely((void **)&this->d_k,
                          sizeof(float) * this->soft_bond_numbers))
    printf("        Error occurs when CUDA malloc BOND_SOFT::d_k in "
           "BOND_SOFT::Memory_Allocate");
  if (!Cuda_Malloc_Safely((void **)&this->d_r0,
                          sizeof(float) * this->soft_bond_numbers))
    printf("        Error occurs when CUDA malloc BOND_SOFT::d_r0 in "
           "BOND_SOFT::Memory_Allocate");
  if (!Cuda_Malloc_Safely((void **)&this->d_ABmask,
                          sizeof(int) * this->soft_bond_numbers))
    printf("         Error occurs when CUDA malloc BOND_SOFT::d_ABmask in "
           "BOND_SOFT::Memory_Allocate");
  if (!Cuda_Malloc_Safely((void **)&this->d_soft_bond_ene,
                          sizeof(float) * this->soft_bond_numbers))
    printf("        Error occurs when CUDA malloc BOND_SOFT::d_bond_ene in "
           "BOND_SOFT::Memory_Allocate");
  if (!Cuda_Malloc_Safely((void **)&this->d_sigma_of_soft_bond_ene,
                          sizeof(float)))
    printf("        Error occurs when CUDA malloc "
           "BOND_SOFT::d_sigma_of_bond_ene in BOND_SOFT::Memory_Allocate");
  if (!Cuda_Malloc_Safely((void **)&this->d_soft_bond_dH_dlambda,
                          sizeof(float) * this->soft_bond_numbers))
    printf("        Error occurs when CUDA malloc "
           "BOND_SOFT::d_soft_bond_dH_dlambda in BOND_SOFT::Memory_Allocate");
  if (!Cuda_Malloc_Safely((void **)&this->d_sigma_of_dH_dlambda, sizeof(float)))
    printf("        Error occurs when CUDA malloc "
           "BOND_SOFT::d_sigma_of_dH_dlambda in BOND_SOFT::Memory_Allocate");
}

void BOND_SOFT::Parameter_Host_To_Device() {
  cudaMemcpy(this->d_atom_a, this->h_atom_a,
             sizeof(int) * this->soft_bond_numbers, cudaMemcpyHostToDevice);
  cudaMemcpy(this->d_atom_b, this->h_atom_b,
             sizeof(int) * this->soft_bond_numbers, cudaMemcpyHostToDevice);
  cudaMemcpy(this->d_k, this->h_k, sizeof(float) * this->soft_bond_numbers,
             cudaMemcpyHostToDevice);
  cudaMemcpy(this->d_r0, this->h_r0, sizeof(float) * this->soft_bond_numbers,
             cudaMemcpyHostToDevice);
  cudaMemcpy(this->d_ABmask, this->h_ABmask,
             sizeof(float) * this->soft_bond_numbers, cudaMemcpyHostToDevice);
  cudaMemcpy(this->d_soft_bond_ene, this->h_soft_bond_ene,
             sizeof(float) * this->soft_bond_numbers, cudaMemcpyHostToDevice);
  cudaMemcpy(this->d_sigma_of_soft_bond_ene, this->h_sigma_of_soft_bond_ene,
             sizeof(float), cudaMemcpyHostToDevice);
}

void BOND_SOFT::Clear() {
  if (is_initialized) {
    cudaFree(this->d_atom_a);
    cudaFree(this->d_atom_b);
    cudaFree(this->d_k);
    cudaFree(this->d_r0);
    cudaFree(this->d_ABmask);
    cudaFree(this->d_soft_bond_ene);
    cudaFree(this->d_sigma_of_soft_bond_ene);

    free(this->h_atom_a);
    free(this->h_atom_b);
    free(this->h_k);
    free(this->h_r0);
    free(this->h_soft_bond_ene);
    free(this->h_sigma_of_soft_bond_ene);

    h_atom_a = NULL;
    d_atom_a = NULL;
    h_atom_b = NULL;
    d_atom_b = NULL;
    d_ABmask = NULL;
    d_k = NULL;
    h_k = NULL;
    d_r0 = NULL;
    h_r0 = NULL;
    h_ABmask = NULL;

    h_soft_bond_ene = NULL;
    d_soft_bond_ene = NULL;
    d_sigma_of_soft_bond_ene = NULL;
    h_sigma_of_soft_bond_ene = NULL;

    is_initialized = 0;
  }
}

void BOND_SOFT::Soft_Bond_Force_With_Atom_Energy_And_Virial(
    const UNSIGNED_INT_VECTOR *uint_crd, const VECTOR scaler, VECTOR *frc,
    float *atom_energy, float *atom_virial) {
  if (is_initialized) {
    Soft_Bond_Force_With_Atom_Energy_And_Virial_CUDA<<<
        (unsigned int)ceilf((float)this->soft_bond_numbers /
                            this->threads_per_block),
        this->threads_per_block>>>(this->soft_bond_numbers, uint_crd, scaler,
                                   this->d_atom_a, this->d_atom_b, this->d_k,
                                   this->d_r0, this->d_ABmask, frc, atom_energy,
                                   atom_virial, this->lambda, this->alpha);
  }
}

float BOND_SOFT::Get_Energy(const UNSIGNED_INT_VECTOR *uint_crd,
                            const VECTOR scaler, int is_download) {
  if (is_initialized) {
    Soft_Bond_Energy_CUDA<<<(unsigned int)ceilf((float)this->soft_bond_numbers /
                                                this->threads_per_block),
                            this->threads_per_block>>>(
        this->soft_bond_numbers, uint_crd, scaler, this->d_atom_a,
        this->d_atom_b, this->d_k, this->d_r0, this->d_ABmask,
        this->d_soft_bond_ene, this->lambda, this->alpha);

    Sum_Of_List<<<1, 1024>>>(this->soft_bond_numbers, this->d_soft_bond_ene,
                             this->d_sigma_of_soft_bond_ene);
    if (is_download) {
      cudaMemcpy(this->h_sigma_of_soft_bond_ene, this->d_sigma_of_soft_bond_ene,
                 sizeof(float), cudaMemcpyDeviceToHost);
      return this->h_sigma_of_soft_bond_ene[0];
    } else {
      return 0;
    }
  }
  return NAN;
}

float BOND_SOFT::Get_Partial_H_Partial_Lambda(
    const UNSIGNED_INT_VECTOR *uint_crd, const VECTOR scaler, int is_download) {
  if (is_initialized) {
    Soft_Bond_dH_dlambda_CUDA<<<(unsigned int)ceilf(
                                    (float)this->soft_bond_numbers /
                                    this->threads_per_block),
                                this->threads_per_block>>>(
        this->soft_bond_numbers, uint_crd, scaler, this->d_atom_a,
        this->d_atom_b, this->d_k, this->d_r0, this->d_ABmask,
        this->d_soft_bond_dH_dlambda, this->lambda, this->alpha);

    Sum_Of_List<<<1, 1024>>>(this->soft_bond_numbers,
                             this->d_soft_bond_dH_dlambda,
                             this->d_sigma_of_dH_dlambda);
    if (is_download) {
      cudaMemcpy(this->h_sigma_of_dH_dlambda, this->d_sigma_of_dH_dlambda,
                 sizeof(float), cudaMemcpyDeviceToHost);
      return this->h_sigma_of_dH_dlambda[0];
    } else {
      return 0.0;
    }
  } else {
    return NAN;
  }
}
