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

#include "nb14.cuh"

#define TINY 1e-10

static __global__ void
Dihedral_14_LJ_Energy(const int dihedral_14_numbers,
                      const UNSIGNED_INT_VECTOR *uint_crd,
                      const VECTOR boxlength, const int *a_14, const int *b_14,
                      const float *lj_A, const float *lj_B, float *ene) {
  int dihedral_14_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (dihedral_14_i < dihedral_14_numbers) {
    int atom_i = a_14[dihedral_14_i];
    int atom_j = b_14[dihedral_14_i];

    UNSIGNED_INT_VECTOR r1 = uint_crd[atom_i];
    UNSIGNED_INT_VECTOR r2 = uint_crd[atom_j];

    int int_x;
    int int_y;
    int int_z;
    VECTOR dr;
    float dr2;
    float dr_2;
    float dr_4;
    float dr_6;
    float dr_12;
    float ene_lin = 0.;

    int_x = r2.uint_x - r1.uint_x;
    int_y = r2.uint_y - r1.uint_y;
    int_z = r2.uint_z - r1.uint_z;
    dr.x = boxlength.x * int_x;
    dr.y = boxlength.y * int_y;
    dr.z = boxlength.z * int_z;
    dr2 = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;

    dr_2 = 1. / dr2;
    dr_4 = dr_2 * dr_2;
    dr_6 = dr_4 * dr_2;
    dr_12 = dr_6 * dr_6;

    ene_lin = 0.08333333 * lj_A[dihedral_14_i] * dr_12 -
              0.1666666 * lj_B[dihedral_14_i] *
                  dr_6; // LJ的A,B系数已经乘以12和6因此要反乘

    ene[dihedral_14_i] = ene_lin;
  }
}

static __global__ void
Dihedral_14_CF_Energy(const int dihedral_14_numbers,
                      const UNSIGNED_INT_VECTOR *uint_crd, const float *charge,
                      const VECTOR boxlength, const int *a_14, const int *b_14,
                      const float *cf_scale_factor, float *ene) {
  int dihedral_14_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (dihedral_14_i < dihedral_14_numbers) {
    int atom_i = a_14[dihedral_14_i];
    int atom_j = b_14[dihedral_14_i];

    UNSIGNED_INT_VECTOR r1 = uint_crd[atom_i];
    UNSIGNED_INT_VECTOR r2 = uint_crd[atom_j];

    int int_x;
    int int_y;
    int int_z;
    VECTOR dr;
    float r_1;
    float ene_lin = 0.;

    int_x = r2.uint_x - r1.uint_x;
    int_y = r2.uint_y - r1.uint_y;
    int_z = r2.uint_z - r1.uint_z;
    dr.x = boxlength.x * int_x;
    dr.y = boxlength.y * int_y;
    dr.z = boxlength.z * int_z;
    r_1 = rnorm3df(dr.x, dr.y, dr.z);

    float charge_i = charge[atom_i];
    if (fabs(charge_i) < TINY)
      charge_i = TINY;
    float charge_j = charge[atom_j];
    if (fabs(charge_j) < TINY)
      charge_j = TINY;
    ene_lin = cf_scale_factor[dihedral_14_i] * charge_i * charge_j * r_1;

    ene[dihedral_14_i] = ene_lin;
  }
}

static __global__ void Dihedral_14_LJ_CF_Force_With_Atom_Energy_And_Virial_Cuda(
    const int dihedral_14_numbers, const UNSIGNED_INT_VECTOR *uint_crd,
    const VECTOR scaler, const int *a_14, const int *b_14,
    const float *cf_scale_factor, const float *charge, const float *lj_A,
    const float *lj_B, VECTOR *frc, float *atom_energy, float *atom_virial) {
  int dihedral_14_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (dihedral_14_i < dihedral_14_numbers) {
    UNSIGNED_INT_VECTOR r1, r2;
    VECTOR dr;
    float dr_abs;
    float dr2;
    float dr_1;
    float dr_2;
    float dr_4;
    float dr_8;
    float dr_14;
    float frc_abs = 0.;
    VECTOR temp_frc;

    float ene_lin;
    float ene_lin2;

    int atom_i = a_14[dihedral_14_i];
    int atom_j = b_14[dihedral_14_i];

    r1 = uint_crd[atom_i];
    r2 = uint_crd[atom_j];

    dr = Get_Periodic_Displacement(r2, r1, scaler);

    dr2 = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;

    dr_2 = 1.0 / dr2;
    dr_4 = dr_2 * dr_2;
    dr_8 = dr_4 * dr_4;
    dr_14 = dr_8 * dr_4 * dr_2;
    dr_abs = norm3df(dr.x, dr.y, dr.z);
    dr_1 = 1. / dr_abs;

    // CF
    float charge_i = charge[atom_i];
    if (fabs(charge_i) < TINY)
      charge_i = TINY;
    float charge_j = charge[atom_j];
    if (fabs(charge_j) < TINY)
      charge_j = TINY;
    float frc_cf_abs;
    frc_cf_abs = cf_scale_factor[dihedral_14_i] * dr_2 * dr_1;
    frc_cf_abs = -frc_cf_abs * charge_i * charge_j;
    // LJ
    frc_abs = -lj_A[dihedral_14_i] * dr_14 + lj_B[dihedral_14_i] * dr_8;

    frc_abs += frc_cf_abs;
    temp_frc.x = frc_abs * dr.x;
    temp_frc.y = frc_abs * dr.y;
    temp_frc.z = frc_abs * dr.z;

    atomicAdd(&frc[atom_j].x, -temp_frc.x);
    atomicAdd(&frc[atom_j].y, -temp_frc.y);
    atomicAdd(&frc[atom_j].z, -temp_frc.z);
    atomicAdd(&frc[atom_i].x, temp_frc.x);
    atomicAdd(&frc[atom_i].y, temp_frc.y);
    atomicAdd(&frc[atom_i].z, temp_frc.z);

    //能量
    ene_lin = cf_scale_factor[dihedral_14_i] * dr_1 * charge_i * charge_j;
    ene_lin2 = 0.08333333 * lj_A[dihedral_14_i] * dr_4 * dr_8 -
               0.1666666 * lj_B[dihedral_14_i] * dr_4 *
                   dr_2; // LJ的A,B系数已经乘以12和6因此要反乘

    atomicAdd(&atom_energy[atom_i], ene_lin + ene_lin2);

    //维里
    atomicAdd(&atom_virial[atom_i], -temp_frc * dr);
  }
}

void NON_BOND_14::Initial(CONTROLLER *controller, const float *LJ_type_A,
                          const float *LJ_type_B, const int *lj_atom_type,
                          const char *module_name) {
  if (module_name == NULL) {
    strcpy(this->module_name, "nb14");
  } else {
    strcpy(this->module_name, module_name);
  }

  char file_name_suffix[CHAR_LENGTH_MAX], file_name_suffix2[CHAR_LENGTH_MAX];
  FILE *fp = NULL, *fp2 = NULL;
  float h_lj_scale_factor = 0;

  sprintf(file_name_suffix, "in_file");
  if (controller[0].Command_Exist(this->module_name, file_name_suffix)) {
    Open_File_Safely(
        &fp, controller[0].Command(this->module_name, file_name_suffix), "r");
    int scanf_ret = fscanf(fp, "%d", &nb14_numbers);
  }

  int extra_numbers = 0;
  sprintf(file_name_suffix2, "extra_in_file");
  if (controller[0].Command_Exist(this->module_name, file_name_suffix2)) {
    Open_File_Safely(
        &fp2, controller[0].Command(this->module_name, file_name_suffix2), "r");
    int scanf_ret = fscanf(fp2, "%d", &extra_numbers);
  }
  nb14_numbers += extra_numbers;
  if (nb14_numbers > 0) {
    controller[0].printf("START INITIALIZING NB14 (%s_%s):\n",
                         this->module_name, file_name_suffix);
    controller[0].printf("    non-bond 14 numbers is %d\n", nb14_numbers);
    Memory_Allocate();
    int smallertype, biggertype, temp;
    for (int i = extra_numbers; i < nb14_numbers; i++) {
      int scanf_ret = fscanf(fp, "%d %d %f %f", h_atom_a + i, h_atom_b + i,
                             &h_lj_scale_factor, h_cf_scale_factor + i);
      smallertype = lj_atom_type[h_atom_a[i]];
      biggertype = lj_atom_type[h_atom_b[i]];
      if (smallertype > biggertype) {
        temp = smallertype;
        smallertype = biggertype;
        biggertype = temp;
      }
      temp = biggertype * (biggertype + 1) / 2 + smallertype;
      h_A[i] = h_lj_scale_factor * LJ_type_A[temp];
      h_B[i] = h_lj_scale_factor * LJ_type_B[temp];
    }
    for (int i = 0; i < extra_numbers; i++) {
      int scanf_ret = fscanf(fp2, "%d %d %f %f %f", h_atom_a + i, h_atom_b + i,
                             h_A + i, h_B + i, h_cf_scale_factor + i);
      h_A[i] *= 12;
      h_B[i] *= 6;
    }
    if (fp != NULL)
      fclose(fp);
    if (fp2 != NULL)
      fclose(fp2);
    Parameter_Host_To_Device();
    is_initialized = 1;
  } else if (controller[0].Command_Exist("amber_parm7")) {
    controller[0].printf("START INITIALIZING NB14 (amber_parm7):\n");
    Read_Information_From_AMBERFILE(controller[0].Command("amber_parm7"),
                                    controller[0], LJ_type_A, LJ_type_B,
                                    lj_atom_type);
    if (nb14_numbers > 0)
      is_initialized = 1;
  } else {
    controller[0].printf("NB14 IS NOT INITIALIZED\n\n");
  }

  if (is_initialized && !is_controller_printf_initialized) {
    controller[0].Step_Print_Initial("nb14_LJ", "%.2f");
    controller[0].Step_Print_Initial("nb14_EE", "%.2f");
    is_controller_printf_initialized = 1;
    controller[0].printf("    structure last modify date is %d\n",
                         last_modify_date);
  }
  if (is_initialized == 1) {
    controller[0].printf("END INITIALIZING NB14\n\n");
  }
}

void NON_BOND_14::Parameter_Host_To_Device() {
  cudaMemcpy(this->d_atom_a, this->h_atom_a, sizeof(int) * this->nb14_numbers,
             cudaMemcpyHostToDevice);
  cudaMemcpy(this->d_atom_b, this->h_atom_b, sizeof(int) * this->nb14_numbers,
             cudaMemcpyHostToDevice);
  cudaMemcpy(this->d_A, this->h_A, sizeof(int) * this->nb14_numbers,
             cudaMemcpyHostToDevice);
  cudaMemcpy(this->d_B, this->h_B, sizeof(int) * this->nb14_numbers,
             cudaMemcpyHostToDevice);
  cudaMemcpy(this->d_cf_scale_factor, this->h_cf_scale_factor,
             sizeof(int) * this->nb14_numbers, cudaMemcpyHostToDevice);
}

void NON_BOND_14::Memory_Allocate() {
  if (!Malloc_Safely((void **)&this->h_atom_a,
                     sizeof(int) * this->nb14_numbers))
    printf("Error occurs when malloc NON_BOND_14::h_atom_a in "
           "NON_BOND_14::Nb14_Initial");
  if (!Malloc_Safely((void **)&this->h_atom_b,
                     sizeof(int) * this->nb14_numbers))
    printf("Error occurs when malloc NON_BOND_14::h_atom_b in "
           "NON_BOND_14::Nb14_Initial");
  if (!Cuda_Malloc_Safely((void **)&this->d_atom_a,
                          sizeof(int) * this->nb14_numbers))
    printf("Error occurs when malloc NON_BOND_14::d_atom_a in "
           "NON_BOND_14::Nb14_Initial");
  if (!Cuda_Malloc_Safely((void **)&this->d_atom_b,
                          sizeof(int) * this->nb14_numbers))
    printf("Error occurs when malloc NON_BOND_14::d_atom_b in "
           "NON_BOND_14::Nb14_Initial");

  if (!Malloc_Safely((void **)&this->h_A, sizeof(float) * this->nb14_numbers))
    printf("Error occurs when malloc NON_BOND_14::h_A in "
           "NON_BOND_14::Nb14_Initial");
  if (!Malloc_Safely((void **)&this->h_B, sizeof(float) * this->nb14_numbers))
    printf("Error occurs when malloc NON_BOND_14::h_B in "
           "NON_BOND_14::Nb14_Initial");
  if (!Malloc_Safely((void **)&this->h_cf_scale_factor,
                     sizeof(float) * this->nb14_numbers))
    printf("Error occurs when malloc NON_BOND_14::h_cf_scale_factor in "
           "NON_BOND_14::Nb14_Initial");
  if (!Cuda_Malloc_Safely((void **)&this->d_A,
                          sizeof(float) * this->nb14_numbers))
    printf("Error occurs when malloc NON_BOND_14::d_A in "
           "NON_BOND_14::Nb14_Initial");
  if (!Cuda_Malloc_Safely((void **)&this->d_B,
                          sizeof(float) * this->nb14_numbers))
    printf("Error occurs when malloc NON_BOND_14::d_B in "
           "NON_BOND_14::Nb14_Initial");
  if (!Cuda_Malloc_Safely((void **)&this->d_cf_scale_factor,
                          sizeof(float) * this->nb14_numbers))
    printf("Error occurs when malloc NON_BOND_14::d_cf_scale_factor in "
           "NON_BOND_14::Nb14_Initial");
  if (!Cuda_Malloc_Safely((void **)&this->d_nb14_energy,
                          sizeof(float) * this->nb14_numbers))
    printf("Error occurs when malloc NON_BOND_14::d_nb14_ene in "
           "NON_BOND_14::Nb14_Initial");
  if (!Cuda_Malloc_Safely((void **)&this->d_nb14_lj_energy_sum, sizeof(float)))
    printf("Error occurs when malloc NON_BOND_14::d_nb14_lj_ene_sum in "
           "NON_BOND_14::Nb14_Initial");
  if (!Cuda_Malloc_Safely((void **)&this->d_nb14_cf_energy_sum, sizeof(float)))
    printf("Error occurs when malloc NON_BOND_14::d_nb14_cf_ene_sum in "
           "NON_BOND_14::Nb14_Initial");
}

void NON_BOND_14::Clear() {
  if (is_initialized) {
    is_initialized = 0;

    free(h_atom_a);
    free(h_atom_b);
    free(h_A);
    free(h_B);
    free(h_cf_scale_factor);

    cudaFree(d_atom_a);
    cudaFree(d_atom_b);
    free(d_A);
    free(d_B);
    cudaFree(d_cf_scale_factor);
    cudaFree(d_nb14_energy);
    cudaFree(d_nb14_lj_energy_sum);
    cudaFree(d_nb14_cf_energy_sum);

    h_atom_a = NULL;
    h_atom_b = NULL;
    h_A = NULL;
    h_B = NULL;
    h_cf_scale_factor = NULL;

    d_atom_a = NULL;
    d_atom_b = NULL;
    d_A = NULL;
    d_B = NULL;
    d_cf_scale_factor = NULL;
    d_nb14_energy = NULL;
    d_nb14_lj_energy_sum = NULL;
    d_nb14_cf_energy_sum = NULL;
  }
}

void NON_BOND_14::Read_Information_From_AMBERFILE(const char *file_name,
                                                  CONTROLLER controller,
                                                  const float *LJ_type_A,
                                                  const float *LJ_type_B,
                                                  const int *lj_atom_type) {
  int dihedral_numbers, dihedral_type_numbers, dihedral_with_hydrogen;
  FILE *parm = NULL;
  Open_File_Safely(&parm, file_name, "r");
  char temps[CHAR_LENGTH_MAX];
  char temp_first_str[CHAR_LENGTH_MAX];
  char temp_second_str[CHAR_LENGTH_MAX];
  int i, tempi, tempi2, tempa, tempb;
  float *lj_scale_type_cpu = NULL, *cf_scale_type_cpu = NULL;
  controller.printf("    Reading non-bond 14 information from AMBER file:\n");
  while (true) {
    if (fgets(temps, CHAR_LENGTH_MAX, parm) == NULL) {
      break;
    }
    if (sscanf(temps, "%s %s", temp_first_str, temp_second_str) != 2) {
      continue;
    }
    if (strcmp(temp_first_str, "%FLAG") == 0 &&
        strcmp(temp_second_str, "POINTERS") == 0) {
      char *get_ret = fgets(temps, CHAR_LENGTH_MAX, parm);

      for (i = 0; i < 6; i++)
        int scanf_ret = fscanf(parm, "%d", &tempi);

      int scanf_ret = fscanf(parm, "%d", &dihedral_with_hydrogen);
      scanf_ret = fscanf(parm, "%d", &dihedral_numbers);
      dihedral_numbers += dihedral_with_hydrogen;

      for (i = 0; i < 9; i++)
        scanf_ret = fscanf(parm, "%d", &tempi);

      scanf_ret = fscanf(parm, "%d", &dihedral_type_numbers);

      nb14_numbers = dihedral_numbers;
      Memory_Allocate();
      nb14_numbers = 0;
      Malloc_Safely((void **)&cf_scale_type_cpu,
                    sizeof(float) * dihedral_type_numbers);
      Malloc_Safely((void **)&lj_scale_type_cpu,
                    sizeof(float) * dihedral_type_numbers);
    }

    if (strcmp(temp_first_str, "%FLAG") == 0 &&
        strcmp(temp_second_str, "SCEE_SCALE_FACTOR") == 0) {
      controller.printf("\t read dihedral 1-4 CF scale factor\n");
      char *get_ret = fgets(temps, CHAR_LENGTH_MAX, parm);
      for (i = 0; i < dihedral_type_numbers; i++) {
        int scanf_ret = fscanf(parm, "%f", &cf_scale_type_cpu[i]);
      }
    }
    if (strcmp(temp_first_str, "%FLAG") == 0 &&
        strcmp(temp_second_str, "SCNB_SCALE_FACTOR") == 0) {
      controller.printf("\t read dihedral 1-4 LJ scale factor\n");
      char *get_ret = fgets(temps, CHAR_LENGTH_MAX, parm);
      for (i = 0; i < dihedral_type_numbers; i++)
        int scanf_ret = fscanf(parm, "%f", &lj_scale_type_cpu[i]);
    }
    if (strcmp(temp_first_str, "%FLAG") == 0 &&
        strcmp(temp_second_str, "DIHEDRALS_INC_HYDROGEN") == 0) {
      float h_lj_scale_factor;
      int smallertype, biggertype, temptype;
      char *get_ret = fgets(temps, CHAR_LENGTH_MAX, parm);
      for (i = 0; i < dihedral_with_hydrogen; i++) {
        int scanf_ret = fscanf(parm, "%d\n", &tempa);
        scanf_ret = fscanf(parm, "%d\n", &tempi);
        scanf_ret = fscanf(parm, "%d\n", &tempi2);
        scanf_ret = fscanf(parm, "%d\n", &tempb);
        scanf_ret = fscanf(parm, "%d\n", &tempi);

        tempi -= 1;
        if (tempi2 > 0) {
          h_atom_a[nb14_numbers] = tempa / 3;
          h_atom_b[nb14_numbers] = abs(tempb / 3);
          h_lj_scale_factor = lj_scale_type_cpu[tempi];

          if (h_lj_scale_factor != 0) {
            h_lj_scale_factor = 1.0f / h_lj_scale_factor;
          }
          h_cf_scale_factor[nb14_numbers] = cf_scale_type_cpu[tempi];
          if (h_cf_scale_factor[nb14_numbers] != 0)
            h_cf_scale_factor[nb14_numbers] =
                1.0f / h_cf_scale_factor[nb14_numbers];

          smallertype = lj_atom_type[h_atom_a[nb14_numbers]];
          biggertype = lj_atom_type[h_atom_b[nb14_numbers]];
          if (smallertype > biggertype) {
            temptype = smallertype;
            smallertype = biggertype;
            biggertype = temptype;
          }
          temptype = biggertype * (biggertype + 1) / 2 + smallertype;
          h_A[nb14_numbers] = h_lj_scale_factor * LJ_type_A[temptype];
          h_B[nb14_numbers] = h_lj_scale_factor * LJ_type_B[temptype];
          nb14_numbers += 1;
        }
      }
    }
    if (strcmp(temp_first_str, "%FLAG") == 0 &&
        strcmp(temp_second_str, "DIHEDRALS_WITHOUT_HYDROGEN") == 0) {
      char *get_ret = fgets(temps, CHAR_LENGTH_MAX, parm);
      float h_lj_scale_factor;
      int smallertype, biggertype, temptype;
      for (i = dihedral_with_hydrogen; i < dihedral_numbers; i++) {
        int scanf_ret = fscanf(parm, "%d\n", &tempa);
        scanf_ret = fscanf(parm, "%d\n", &tempi);
        scanf_ret = fscanf(parm, "%d\n", &tempi2);
        scanf_ret = fscanf(parm, "%d\n", &tempb);
        scanf_ret = fscanf(parm, "%d\n", &tempi);

        tempi -= 1;
        if (tempi2 > 0) {
          h_atom_a[nb14_numbers] = tempa / 3;
          h_atom_b[nb14_numbers] = abs(tempb / 3);
          h_lj_scale_factor = lj_scale_type_cpu[tempi];

          if (h_lj_scale_factor != 0) {
            h_lj_scale_factor = 1.0f / h_lj_scale_factor;
          }

          smallertype = lj_atom_type[h_atom_a[nb14_numbers]];
          biggertype = lj_atom_type[h_atom_b[nb14_numbers]];
          if (smallertype > biggertype) {
            temptype = smallertype;
            smallertype = biggertype;
            biggertype = temptype;
          }
          temptype = biggertype * (biggertype + 1) / 2 + smallertype;
          h_A[nb14_numbers] = h_lj_scale_factor * LJ_type_A[temptype];
          h_B[nb14_numbers] = h_lj_scale_factor * LJ_type_B[temptype];

          h_cf_scale_factor[nb14_numbers] = cf_scale_type_cpu[tempi];
          if (h_cf_scale_factor[nb14_numbers] != 0)
            h_cf_scale_factor[nb14_numbers] =
                1.0f / h_cf_scale_factor[nb14_numbers];
          nb14_numbers += 1;
        }
      }
    }
  }

  free(lj_scale_type_cpu);
  free(cf_scale_type_cpu);
  fclose(parm);
  controller.printf("        nb14_number is %d\n", nb14_numbers);
  controller.printf("    End reading nb14 information from AMBER file\n");
  Parameter_Host_To_Device();
}

float NON_BOND_14::Get_14_LJ_Energy(const UNSIGNED_INT_VECTOR *uint_crd,
                                    const VECTOR scaler, int is_download) {
  if (is_initialized) {
    Dihedral_14_LJ_Energy<<<(unsigned int)ceilf((float)nb14_numbers /
                                                threads_per_block),
                            threads_per_block>>>(nb14_numbers, uint_crd, scaler,
                                                 d_atom_a, d_atom_b, d_A, d_B,
                                                 d_nb14_energy);
    Sum_Of_List<<<1, 1024>>>(nb14_numbers, d_nb14_energy, d_nb14_lj_energy_sum);
    if (is_download) {
      cudaMemcpy(&h_nb14_lj_energy_sum, this->d_nb14_lj_energy_sum,
                 sizeof(float), cudaMemcpyDeviceToHost);
      return h_nb14_lj_energy_sum;
    } else {
      return 0;
    }
  }
  return NAN;
}

float NON_BOND_14::Get_14_CF_Energy(const UNSIGNED_INT_VECTOR *uint_crd,
                                    const float *charge, const VECTOR scaler,
                                    int is_download) {
  if (is_initialized) {
    Dihedral_14_CF_Energy<<<
        (unsigned int)ceilf((float)nb14_numbers / threads_per_block),
        threads_per_block>>>(nb14_numbers, uint_crd, charge, scaler, d_atom_a,
                             d_atom_b, d_cf_scale_factor, d_nb14_energy);
    Sum_Of_List<<<1, 1024>>>(nb14_numbers, d_nb14_energy, d_nb14_cf_energy_sum);
    if (is_download) {
      cudaMemcpy(&h_nb14_cf_energy_sum, this->d_nb14_cf_energy_sum,
                 sizeof(float), cudaMemcpyDeviceToHost);
      return h_nb14_cf_energy_sum;
    } else {
      return 0;
    }
  }
  return NAN;
}

void NON_BOND_14::Non_Bond_14_LJ_CF_Force_With_Atom_Energy_And_Virial(
    const UNSIGNED_INT_VECTOR *uint_crd, const float *charge,
    const VECTOR scaler, VECTOR *frc, float *atom_energy, float *atom_virial) {
  if (is_initialized) {
    Dihedral_14_LJ_CF_Force_With_Atom_Energy_And_Virial_Cuda<<<
        (unsigned int)ceilf((float)nb14_numbers / threads_per_block),
        threads_per_block>>>(nb14_numbers, uint_crd, scaler, d_atom_a, d_atom_b,
                             d_cf_scale_factor, charge, d_A, d_B, frc,
                             atom_energy, atom_virial);
  }
}
