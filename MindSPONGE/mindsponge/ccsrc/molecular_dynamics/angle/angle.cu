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

#include "angle.cuh"

// the formula is deduced by xyj
static __global__ void
Angle_Energy_CUDA(const int angle_numbers, const UNSIGNED_INT_VECTOR *uint_crd,
                  const VECTOR scaler, const int *atom_a, const int *atom_b,
                  const int *atom_c, const float *angle_k,
                  const float *angle_theta0, float *angle_energy) {
  int angle_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (angle_i < angle_numbers) {
    int atom_i = atom_a[angle_i];
    int atom_j = atom_b[angle_i];
    int atom_k = atom_c[angle_i];

    float theta0 = angle_theta0[angle_i];
    float k = angle_k[angle_i];

    VECTOR drij =
        Get_Periodic_Displacement(uint_crd[atom_i], uint_crd[atom_j], scaler);
    VECTOR drkj =
        Get_Periodic_Displacement(uint_crd[atom_k], uint_crd[atom_j], scaler);

    float rij_2 = 1. / (drij * drij);
    float rkj_2 = 1. / (drkj * drkj);
    float rij_1_rkj_1 = sqrtf(rij_2 * rkj_2);

    float costheta = drij * drkj * rij_1_rkj_1;
    costheta = fmaxf(-0.999999, fminf(costheta, 0.999999));
    float theta = acosf(costheta);

    float dtheta = theta - theta0;

    angle_energy[angle_i] = k * dtheta * dtheta;
  }
}

__global__ void Angle_Force_With_Atom_Energy_CUDA(
    const int angle_numbers, const UNSIGNED_INT_VECTOR *uint_crd,
    const VECTOR scaler, const int *atom_a, const int *atom_b,
    const int *atom_c, const float *angle_k, const float *angle_theta0,
    VECTOR *frc, float *atom_energy) {
  int angle_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (angle_i < angle_numbers) {
    int atom_i = atom_a[angle_i];
    int atom_j = atom_b[angle_i];
    int atom_k = atom_c[angle_i];

    float theta0 = angle_theta0[angle_i];
    float k = angle_k[angle_i];
    float k2 = k; //复制一份k

    VECTOR drij =
        Get_Periodic_Displacement(uint_crd[atom_i], uint_crd[atom_j], scaler);
    VECTOR drkj =
        Get_Periodic_Displacement(uint_crd[atom_k], uint_crd[atom_j], scaler);

    float rij_2 = 1. / (drij * drij);
    float rkj_2 = 1. / (drkj * drkj);
    float rij_1_rkj_1 = sqrtf(rij_2 * rkj_2);

    float costheta = drij * drkj * rij_1_rkj_1;
    costheta = fmaxf(-0.999999, fminf(costheta, 0.999999));
    float theta = acosf(costheta);

    float dtheta = theta - theta0;
    k = -2 * k * dtheta / sinf(theta);

    float common_factor_cross = k * rij_1_rkj_1;
    float common_factor_self = k * costheta;

    VECTOR fi = common_factor_self * rij_2 * drij - common_factor_cross * drkj;
    VECTOR fk = common_factor_self * rkj_2 * drkj - common_factor_cross * drij;

    atomicAdd(&frc[atom_i].x, fi.x);
    atomicAdd(&frc[atom_i].y, fi.y);
    atomicAdd(&frc[atom_i].z, fi.z);

    atomicAdd(&frc[atom_k].x, fk.x);
    atomicAdd(&frc[atom_k].y, fk.y);
    atomicAdd(&frc[atom_k].z, fk.z);

    fi = -fi - fk;

    atomicAdd(&frc[atom_j].x, fi.x);
    atomicAdd(&frc[atom_j].y, fi.y);
    atomicAdd(&frc[atom_j].z, fi.z);

    atomicAdd(
        &atom_energy[atom_i],
        k2 * dtheta *
            dtheta); //将这个angle的能量加到参与angle的第一个原子上，用直接能量算法得到的能量是不能分解到单个原子上的。
  }
}

void ANGLE::Initial(CONTROLLER *controller, const char *module_name) {
  if (module_name == NULL) {
    strcpy(this->module_name, "angle");
  } else {
    strcpy(this->module_name, module_name);
  }

  char file_name_suffix[CHAR_LENGTH_MAX];
  sprintf(file_name_suffix, "in_file");

  if (controller[0].Command_Exist(this->module_name, file_name_suffix)) {
    controller[0].printf("START INITIALIZING ANGLE (%s_%s):\n",
                         this->module_name, file_name_suffix);
    FILE *fp = NULL;
    Open_File_Safely(&fp, controller[0].Command(this->module_name, "in_file"),
                     "r");

    int scanf_ret = fscanf(fp, "%d", &angle_numbers);
    controller[0].printf("    angle_numbers is %d\n", angle_numbers);
    Memory_Allocate();
    for (int i = 0; i < angle_numbers; i++) {
      int scanf_ret = fscanf(fp, "%d %d %d %f %f", h_atom_a + i, h_atom_b + i,
                             h_atom_c + i, h_angle_k + i, h_angle_theta0 + i);
    }
    fclose(fp);
    Parameter_Host_To_Device();
    is_initialized = 1;
  } else if (controller[0].Command_Exist("amber_parm7")) {
    controller[0].printf("START INITIALIZING ANGLE (amber_parm7):\n");
    Read_Information_From_AMBERFILE(controller[0].Command("amber_parm7"),
                                    controller[0]);
    if (angle_numbers > 0)
      is_initialized = 1;
  } else {
    controller[0].printf("ANGLE IS NOT INITIALIZED\n\n");
  }
  if (is_initialized && !is_controller_printf_initialized) {
    controller[0].Step_Print_Initial(this->module_name, "%.2f");
    is_controller_printf_initialized = 1;
    controller[0].printf("    structure last modify date is %d\n",
                         last_modify_date);
  }
  if (is_initialized) {
    controller[0].printf("END INITIALIZING ANGLE\n\n");
  }
}

void ANGLE::Read_Information_From_AMBERFILE(const char *file_name,
                                            CONTROLLER controller) {
  FILE *parm = NULL;
  Open_File_Safely(&parm, file_name, "r");
  int angle_with_H_numbers = 0;
  int angle_without_H_numbers = 0;
  int angle_count = 0;

  int angle_type_numbers = 0;
  float *type_k = NULL, *type_theta0 = NULL;
  int *h_type = NULL;

  controller.printf("    Reading angle information from AMBER file:\n");

  char temps[CHAR_LENGTH_MAX];
  char temp_first_str[CHAR_LENGTH_MAX];
  char temp_second_str[CHAR_LENGTH_MAX];

  while (true) {
    if (!fgets(temps, CHAR_LENGTH_MAX, parm)) {
      break;
    }
    if (sscanf(temps, "%s %s", temp_first_str, temp_second_str) != 2) {
      continue;
    }

    if (strcmp(temp_first_str, "%FLAG") == 0 &&
        strcmp(temp_second_str, "POINTERS") == 0) {
      char *get_ret = fgets(temps, CHAR_LENGTH_MAX, parm);
      int lin;
      for (int i = 0; i < 4; i = i + 1) {
        int scanf_ret = fscanf(parm, "%d", &lin);
      }
      int scanf_ret = fscanf(parm, "%d", &angle_with_H_numbers);
      scanf_ret = fscanf(parm, "%d", &angle_without_H_numbers);
      this->angle_numbers = angle_with_H_numbers + angle_without_H_numbers;
      controller.printf("        angle_numbers is %d\n", this->angle_numbers);

      this->Memory_Allocate();

      for (int i = 0; i < 10; i = i + 1) {
        scanf_ret = fscanf(parm, "%d", &lin);
      }
      scanf_ret = fscanf(parm, "%d", &angle_type_numbers);
      controller.printf("        angle_type_numbers is %d\n",
                        angle_type_numbers);

      if (!Malloc_Safely((void **)&h_type, sizeof(int) * this->angle_numbers)) {
        controller.printf("        Error occurs when malloc h_type in "
                          "ANGLE::Read_Information_From_AMBERFILE");
      }

      if (!Malloc_Safely((void **)&type_k,
                         sizeof(float) * angle_type_numbers)) {
        controller.printf("        Error occurs when malloc type_k in "
                          "ANGLE::Read_Information_From_AMBERFILE");
      }
      if (!Malloc_Safely((void **)&type_theta0,
                         sizeof(float) * angle_type_numbers)) {
        controller.printf("        Error occurs when malloc type_theta0 in "
                          "ANGLE::Read_Information_From_AMBERFILE");
      }
    } // POINTER

    // read angle type
    if (strcmp(temp_first_str, "%FLAG") == 0 &&
        strcmp(temp_second_str, "ANGLES_INC_HYDROGEN") == 0) {
      controller.printf("        reading angle_with_hydrogen %d\n",
                        angle_with_H_numbers);
      char *get_ret = fgets(temps, CHAR_LENGTH_MAX, parm);
      for (int i = 0; i < angle_with_H_numbers; i = i + 1) {
        int scanf_ret = fscanf(parm, "%d\n", &this->h_atom_a[angle_count]);
        scanf_ret = fscanf(parm, "%d\n", &this->h_atom_b[angle_count]);
        scanf_ret = fscanf(parm, "%d\n", &this->h_atom_c[angle_count]);
        scanf_ret = fscanf(parm, "%d\n", &h_type[angle_count]);
        this->h_atom_a[angle_count] = this->h_atom_a[angle_count] / 3;
        this->h_atom_b[angle_count] = this->h_atom_b[angle_count] / 3;
        this->h_atom_c[angle_count] = this->h_atom_c[angle_count] / 3;
        h_type[angle_count] = h_type[angle_count] - 1;
        angle_count = angle_count + 1;
      }
    } // angle type
    if (strcmp(temp_first_str, "%FLAG") == 0 &&
        strcmp(temp_second_str, "ANGLES_WITHOUT_HYDROGEN") == 0) {
      controller.printf("        reading angle_without_hydrogen %d\n",
                        angle_without_H_numbers);
      char *get_ret = fgets(temps, CHAR_LENGTH_MAX, parm);
      for (int i = 0; i < angle_without_H_numbers; i = i + 1) {
        int scanf_ret = fscanf(parm, "%d\n", &this->h_atom_a[angle_count]);
        scanf_ret = fscanf(parm, "%d\n", &this->h_atom_b[angle_count]);
        scanf_ret = fscanf(parm, "%d\n", &this->h_atom_c[angle_count]);
        scanf_ret = fscanf(parm, "%d\n", &h_type[angle_count]);
        this->h_atom_a[angle_count] = this->h_atom_a[angle_count] / 3;
        this->h_atom_b[angle_count] = this->h_atom_b[angle_count] / 3;
        this->h_atom_c[angle_count] = this->h_atom_c[angle_count] / 3;
        h_type[angle_count] = h_type[angle_count] - 1;
        angle_count = angle_count + 1;
      }
    }
    if (strcmp(temp_first_str, "%FLAG") == 0 &&
        strcmp(temp_second_str, "ANGLE_FORCE_CONSTANT") == 0) {
      char *scanf_ret = fgets(temps, CHAR_LENGTH_MAX, parm);
      for (int i = 0; i < angle_type_numbers; i = i + 1) {
        int scanf_ret = fscanf(parm, "%f\n", &type_k[i]);
      }
    }
    if (strcmp(temp_first_str, "%FLAG") == 0 &&
        strcmp(temp_second_str, "ANGLE_EQUIL_VALUE") == 0) {
      char *scanf_ret = fgets(temps, CHAR_LENGTH_MAX, parm);
      for (int i = 0; i < angle_type_numbers; i = i + 1) {
        int scanf_ret = fscanf(parm, "%f\n", &type_theta0[i]); // in rad
      }
    }
  } // while
  if (this->angle_numbers != angle_count) {
    controller.printf("        angle_count %d!= angle_numbers %d!\n",
                      angle_count, this->angle_numbers);
    getchar();
  }
  for (int i = 0; i < this->angle_numbers; i = i + 1) {
    this->h_angle_k[i] = type_k[h_type[i]];
    this->h_angle_theta0[i] = type_theta0[h_type[i]];
  }

  controller.printf("    End reading angle information from AMBER file\n");
  fclose(parm);
  free(h_type);
  free(type_k);
  free(type_theta0);

  Parameter_Host_To_Device();
  is_initialized = 1;
  if (angle_numbers == 0)
    Clear();
}

void ANGLE::Memory_Allocate() {
  if (!Malloc_Safely((void **)&(this->h_atom_a),
                     sizeof(int) * this->angle_numbers))
    printf("        Error occurs when malloc ANGLE::h_atom_a in "
           "ANGLE::Angle_Initialize");
  if (!Malloc_Safely((void **)&(this->h_atom_b),
                     sizeof(int) * this->angle_numbers))
    printf("        Error occurs when malloc ANGLE::h_atom_b in "
           "ANGLE::Angle_Initialize");
  if (!Malloc_Safely((void **)&(this->h_atom_c),
                     sizeof(int) * this->angle_numbers))
    printf("        Error occurs when malloc ANGLE::h_atom_c in "
           "ANGLE::Angle_Initialize");
  if (!Malloc_Safely((void **)&(this->h_angle_k),
                     sizeof(float) * this->angle_numbers))
    printf("        Error occurs when malloc ANGLE::h_angle_k in "
           "ANGLE::Angle_Initialize");
  if (!Malloc_Safely((void **)&(this->h_angle_theta0),
                     sizeof(float) * this->angle_numbers))
    printf("        Error occurs when malloc ANGLE::h_angle_theta0 in "
           "ANGLE::Angle_Initialize");
  if (!Malloc_Safely((void **)&(this->h_angle_ene),
                     sizeof(float) * this->angle_numbers))
    printf("        Error occurs when malloc ANGLE::h_angle_ene in "
           "ANGLE::Angle_Initialize");
  if (!Malloc_Safely((void **)&(this->h_sigma_of_angle_ene), sizeof(float)))
    printf("        Error occurs when malloc ANGLE::h_sigma_of_angle_ene in "
           "ANGLE::Angle_Initialize");

  if (!Cuda_Malloc_Safely((void **)&this->d_atom_a,
                          sizeof(int) * this->angle_numbers))
    printf("        Error occurs when CUDA malloc ANGLE::d_atom_a in "
           "ANGLE::Angle_Initialize");
  if (!Cuda_Malloc_Safely((void **)&this->d_atom_b,
                          sizeof(int) * this->angle_numbers))
    printf("        Error occurs when CUDA malloc ANGLE::d_atom_b in "
           "ANGLE::Angle_Initialize");
  if (!Cuda_Malloc_Safely((void **)&this->d_atom_c,
                          sizeof(int) * this->angle_numbers))
    printf("        Error occurs when CUDA malloc ANGLE::d_atom_c in "
           "ANGLE::Angle_Initialize");
  if (!Cuda_Malloc_Safely((void **)&this->d_angle_k,
                          sizeof(float) * this->angle_numbers))
    printf("        Error occurs when CUDA malloc ANGLE::d_angle_k in "
           "ANGLE::Angle_Initialize");
  if (!Cuda_Malloc_Safely((void **)&this->d_angle_theta0,
                          sizeof(float) * this->angle_numbers))
    printf("        Error occurs when CUDA malloc ANGLE::d_angle_theta0 in "
           "ANGLE::Angle_Initialize");
  if (!Cuda_Malloc_Safely((void **)&this->d_angle_ene,
                          sizeof(float) * this->angle_numbers))
    printf("        Error occurs when CUDA malloc ANGLE::d_angle_ene in "
           "ANGLE::Angle_Initialize");
  if (!Cuda_Malloc_Safely((void **)&this->d_sigma_of_angle_ene, sizeof(float)))
    printf("        Error occurs when CUDA malloc ANGLE::d_sigma_of_angle_ene "
           "in ANGLE::Angle_Initialize");
}

void ANGLE::Parameter_Host_To_Device() {
  cudaMemcpy(this->d_atom_a, this->h_atom_a, sizeof(int) * this->angle_numbers,
             cudaMemcpyHostToDevice);
  cudaMemcpy(this->d_atom_b, this->h_atom_b, sizeof(int) * this->angle_numbers,
             cudaMemcpyHostToDevice);
  cudaMemcpy(this->d_atom_c, this->h_atom_c, sizeof(int) * this->angle_numbers,
             cudaMemcpyHostToDevice);
  cudaMemcpy(this->d_angle_k, this->h_angle_k,
             sizeof(float) * this->angle_numbers, cudaMemcpyHostToDevice);
  cudaMemcpy(this->d_angle_theta0, this->h_angle_theta0,
             sizeof(float) * this->angle_numbers, cudaMemcpyHostToDevice);
  cudaMemcpy(this->d_angle_ene, this->h_angle_ene,
             sizeof(float) * this->angle_numbers, cudaMemcpyHostToDevice);
  cudaMemcpy(this->d_sigma_of_angle_ene, this->h_sigma_of_angle_ene,
             sizeof(float), cudaMemcpyHostToDevice);
}

void ANGLE::Clear() {
  if (is_initialized) {
    is_initialized = 0;
    free(h_atom_a);
    free(h_atom_b);
    free(h_atom_c);
    free(h_angle_k);
    free(h_angle_theta0);
    free(h_angle_ene);
    free(h_sigma_of_angle_ene);

    cudaFree(d_atom_a);
    cudaFree(d_atom_b);
    cudaFree(d_atom_c);
    cudaFree(d_angle_k);
    cudaFree(d_angle_theta0);
    cudaFree(d_angle_ene);
    cudaFree(d_sigma_of_angle_ene);

    h_atom_a = NULL;
    h_atom_b = NULL;
    h_atom_c = NULL;
    h_angle_k = NULL;
    h_angle_theta0 = NULL;
    h_angle_ene = NULL;
    h_sigma_of_angle_ene = NULL;
    d_atom_a = NULL;
    d_atom_b = NULL;
    d_atom_c = NULL;
    d_angle_k = NULL;
    d_angle_theta0 = NULL;
    d_angle_ene = NULL;
    d_sigma_of_angle_ene = NULL;
  }
}

float ANGLE::Get_Energy(const UNSIGNED_INT_VECTOR *uint_crd,
                        const VECTOR scaler, int is_download) {
  if (is_initialized) {
    Angle_Energy_CUDA<<<(unsigned int)ceilf((float)this->angle_numbers /
                                            this->threads_per_block),
                        this->threads_per_block>>>(
        this->angle_numbers, uint_crd, scaler, this->d_atom_a, this->d_atom_b,
        this->d_atom_c, this->d_angle_k, this->d_angle_theta0,
        this->d_angle_ene);
    Sum_Of_List<<<1, 1024>>>(this->angle_numbers, this->d_angle_ene,
                             this->d_sigma_of_angle_ene);
    if (is_download) {
      cudaMemcpy(this->h_sigma_of_angle_ene, this->d_sigma_of_angle_ene,
                 sizeof(float), cudaMemcpyDeviceToHost);
      return h_sigma_of_angle_ene[0];
    } else {
      return 0;
    }
  }
  return NAN;
}

void ANGLE::Angle_Force_With_Atom_Energy(const UNSIGNED_INT_VECTOR *uint_crd,
                                         const VECTOR scaler, VECTOR *frc,
                                         float *atom_energy) {
  if (is_initialized) {
    Angle_Force_With_Atom_Energy_CUDA<<<(unsigned int)ceilf(
                                            (float)this->angle_numbers /
                                            this->threads_per_block),
                                        this->threads_per_block>>>(
        this->angle_numbers, uint_crd, scaler, this->d_atom_a, this->d_atom_b,
        this->d_atom_c, this->d_angle_k, this->d_angle_theta0, frc,
        atom_energy);
  }
}
