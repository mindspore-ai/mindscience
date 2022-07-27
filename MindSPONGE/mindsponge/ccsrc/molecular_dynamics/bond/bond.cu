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

#include "bond.cuh"
//使用bond的能量定义，对bond_numbers中的每根bond，都计算目前uint_crd,scaler下的能量并存入对应的bond_ene数组中
static __global__ void Bond_Energy_CUDA(const int bond_numbers,
                                        const UNSIGNED_INT_VECTOR *uint_crd,
                                        const VECTOR scaler, const int *atom_a,
                                        const int *atom_b, const float *bond_k,
                                        const float *bond_r0, float *bond_ene) {
  int bond_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (bond_i < bond_numbers) {
    int atom_i = atom_a[bond_i];
    int atom_j = atom_b[bond_i];

    float k = bond_k[bond_i];
    float r0 = bond_r0[bond_i];

    VECTOR dr =
        Get_Periodic_Displacement(uint_crd[atom_i], uint_crd[atom_j], scaler);

    float r1 = norm3df(dr.x, dr.y, dr.z);
    float tempf = r1 - r0;

    bond_ene[bond_i] = k * tempf * tempf;
  }
}

//由于，大部分情况下bond的energy和virial计算耗时不显著，为简化bond模块的逻辑复杂度，
//将bond
//对原子上的力（frc）、原子上的能量（atom_energy）、原子上的维力值（atom_virial）
//一并计算。
//对于简易和轻度修改，可以不用考虑能量与维力值的计算。
//  在不使用涉及维力系数的模拟中，可以不用计算正确的维力值
//  在能量数值不影响模拟的过程中，可以不用计算正确的能量值
//  只有力是最基本的计算要求
static __global__ void Bond_Force_With_Atom_Energy_And_Virial_CUDA(
    const int bond_numbers, const UNSIGNED_INT_VECTOR *uint_crd,
    const VECTOR scaler, const int *atom_a, const int *atom_b,
    const float *bond_k, const float *bond_r0, VECTOR *frc, float *atom_energy,
    float *atom_virial) {
  int bond_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (bond_i < bond_numbers) {
    //获取第bond_i根键的两个连接的原子编号
    //和键强度、平衡长度
    int atom_i = atom_a[bond_i];
    int atom_j = atom_b[bond_i];
    float k = bond_k[bond_i];
    float r0 = bond_r0[bond_i];

    //获取该对原子的考虑周期性边界的最短位置矢量（dr），和最短距离abs_r
    VECTOR dr =
        Get_Periodic_Displacement(uint_crd[atom_i], uint_crd[atom_j], scaler);
    float abs_r = norm3df(dr.x, dr.y, dr.z);
    float r_1 = 1. / abs_r;

    float tempf2 = abs_r - r0;
    float tempf = 2 * tempf2 * k;
    VECTOR f = tempf * r_1 * dr;

    //将计算得到的力加到对应的原子身上
    atomicAdd(&frc[atom_i].x, -f.x);
    atomicAdd(&frc[atom_i].y, -f.y);
    atomicAdd(&frc[atom_i].z, -f.z);

    atomicAdd(&frc[atom_j].x, f.x);
    atomicAdd(&frc[atom_j].y, f.y);
    atomicAdd(&frc[atom_j].z, f.z);

    //将计算得到的能量和维力值加到该bond中的其中一个原子身上
    //原理上，该bond能量是不可分的。但是，一般情况，bond相连的两个原子
    //总是被看作属于一个分子来讨论，因此可以直接将能量和维力值加到其中一个原子上
    atomicAdd(&atom_virial[atom_i], -tempf * abs_r);
    atomicAdd(&atom_energy[atom_i], k * tempf2 * tempf2);
  }
}

void BOND::Initial(CONTROLLER *controller, const char *module_name) {
  //给予bond模块一个默认名字：bond
  if (module_name == NULL) {
    strcpy(this->module_name, "bond");
  } else {
    strcpy(this->module_name, module_name);
  }

  //指定读入bond信息的后缀，默认为in_file，即合成为bond_in_file
  char file_name_suffix[CHAR_LENGTH_MAX];
  sprintf(file_name_suffix, "in_file");

  //不同的bond初始化方案
  //从bond_in_file对应的文件中读取bond参数
  if (controller[0].Command_Exist(this->module_name, file_name_suffix)) {
    controller[0].printf("START INITIALIZING BOND (%s_%s):\n",
                         this->module_name, file_name_suffix);
    FILE *fp = NULL;
    Open_File_Safely(
        &fp, controller[0].Command(this->module_name, file_name_suffix), "r");

    int read_ret = fscanf(fp, "%d", &bond_numbers);
    controller[0].printf("    bond_numbers is %d\n", bond_numbers);
    Memory_Allocate();
    for (int i = 0; i < bond_numbers; i++) {
      read_ret = fscanf(fp, "%d %d %f %f", h_atom_a + i, h_atom_b + i, h_k + i,
                        h_r0 + i);
    }
    fclose(fp);
    Parameter_Host_To_Device();
    is_initialized = 1;
  } else if (controller[0].Command_Exist("amber_parm7")) {
    controller[0].printf("START INITIALIZING BOND (amber_parm7):\n");
    Read_Information_From_AMBERFILE(controller[0].Command("amber_parm7"),
                                    controller[0]);
    if (bond_numbers > 0)
      is_initialized = 1;
  } else {
    controller[0].printf("BOND IS NOT INITIALIZED\n\n");
  }

  //初始化了，且第一次加载用于间隔输出的信息
  if (is_initialized && !is_controller_printf_initialized) {
    controller[0].Step_Print_Initial(this->module_name, "%.2f");
    is_controller_printf_initialized = 1;
    controller[0].printf("    structure last modify date is %d\n",
                         last_modify_date);
  }

  //初始化完成
  if (is_initialized) {
    controller[0].printf("END INITIALIZING BOND\n\n");
  }
}
void BOND::Read_Information_From_AMBERFILE(const char *file_name,
                                           CONTROLLER controller) {
  float *bond_type_k = NULL;
  float *bond_type_r = NULL;
  int bond_type_numbers = 0;
  FILE *parm = NULL;
  Open_File_Safely(&parm, file_name, "r");

  controller.printf("    Reading bond information from AMBER file:\n");

  char temps[CHAR_LENGTH_MAX];
  char temp_first_str[CHAR_LENGTH_MAX];
  char temp_second_str[CHAR_LENGTH_MAX];
  int i, tempi, bond_with_hydrogen;

  while (true) {
    if (fgets(temps, CHAR_LENGTH_MAX, parm) == NULL) {
      break;
    }
    if (sscanf(temps, "%s %s", temp_first_str, temp_second_str) != 2) {
      continue;
    }
    if (strcmp(temp_first_str, "%FLAG") == 0 &&
        strcmp(temp_second_str, "POINTERS") == 0) {
      //读取parm7中的这一行“%FORMAT(10I8)”
      char *read_ret = fgets(temps, CHAR_LENGTH_MAX, parm);

      //前两个和bond信息无关
      for (i = 0; i < 2; i++)
        int read_ret = fscanf(parm, "%d", &tempi);

      int scan_ret = fscanf(parm, "%d", &bond_with_hydrogen);
      scan_ret = fscanf(parm, "%d", &(this->bond_numbers));
      this->bond_numbers += bond_with_hydrogen;

      controller.printf("        bond_numbers is %d\n", this->bond_numbers);

      this->Memory_Allocate();

      //跳过11个记录的无关变量
      for (i = 0; i < 11; i++)
        scan_ret = fscanf(parm, "%d", &tempi);

      scan_ret = fscanf(parm, "%d", &bond_type_numbers);
      controller.printf("        bond_type_numbers is %d\n", bond_type_numbers);

      if (!Malloc_Safely((void **)&bond_type_k,
                         sizeof(float) * bond_type_numbers)) {
        controller.printf("        Error occurs when malloc bond_type_k in "
                          "BOND::Read_Information_From_AMBERFILE");
      }

      if (!Malloc_Safely((void **)&bond_type_r,
                         sizeof(float) * bond_type_numbers)) {
        controller.printf("        Error occurs when malloc bond_type_r in "
                          "BOND::Read_Information_From_AMBERFILE");
      }
    }

    if (strcmp(temp_first_str, "%FLAG") == 0 &&
        strcmp(temp_second_str, "BOND_FORCE_CONSTANT") == 0) {
      controller.printf("        reading bond_type_numbers %d\n",
                        bond_type_numbers);
      char *get_ret = fgets(temps, CHAR_LENGTH_MAX, parm);
      for (i = 0; i < bond_type_numbers; i++) {
        int scan_ret = fscanf(parm, "%f", &bond_type_k[i]);
      }
    }

    if (strcmp(temp_first_str, "%FLAG") == 0 &&
        strcmp(temp_second_str, "BOND_EQUIL_VALUE") == 0) {
      char *get_ret = fgets(temps, CHAR_LENGTH_MAX, parm);
      for (i = 0; i < bond_type_numbers; i++)
        int scan_ret = fscanf(parm, "%f", &bond_type_r[i]);
    }

    if (strcmp(temp_first_str, "%FLAG") == 0 &&
        strcmp(temp_second_str, "BONDS_INC_HYDROGEN") == 0) {
      controller.printf("        reading bond_with_hydrogen %d\n",
                        bond_with_hydrogen);
      char *get_ret = fgets(temps, CHAR_LENGTH_MAX, parm);
      for (i = 0; i < bond_with_hydrogen; i++) {
        int scan_ret = fscanf(parm, "%d\n", &this->h_atom_a[i]);
        scan_ret = fscanf(parm, "%d\n", &this->h_atom_b[i]);
        scan_ret = fscanf(parm, "%d\n", &tempi);
        this->h_atom_a[i] /= 3; // AMBER 上存储的bond的原子编号要除以空间维度
        this->h_atom_b[i] /= 3;
        tempi -= 1;
        this->h_k[i] = bond_type_k[tempi];
        this->h_r0[i] = bond_type_r[tempi];
      }
    }

    if (strcmp(temp_first_str, "%FLAG") == 0 &&
        strcmp(temp_second_str, "BONDS_WITHOUT_HYDROGEN") == 0) {
      controller.printf("        reading bond_without_hydrogen %d\n",
                        this->bond_numbers - bond_with_hydrogen);
      char *get_ret = fgets(temps, CHAR_LENGTH_MAX, parm);
      for (i = bond_with_hydrogen; i < this->bond_numbers; i++) {
        int scan_ret = fscanf(parm, "%d\n", &this->h_atom_a[i]);
        scan_ret = fscanf(parm, "%d\n", &this->h_atom_b[i]);
        scan_ret = fscanf(parm, "%d\n", &tempi);
        this->h_atom_a[i] /= 3;
        this->h_atom_b[i] /= 3;
        tempi -= 1;
        this->h_k[i] = bond_type_k[tempi];
        this->h_r0[i] = bond_type_r[tempi];
      }
    }
  }
  controller.printf("    End reading bond information from AMBER file\n");
  fclose(parm);

  free(bond_type_k);
  free(bond_type_r);

  Parameter_Host_To_Device();
  is_initialized = 1;
  if (bond_numbers == 0) {
    Clear();
  }
}

void BOND::Memory_Allocate() {
  if (!Malloc_Safely((void **)&(this->h_atom_a),
                     sizeof(int) * this->bond_numbers))
    printf("        Error occurs when malloc BOND::h_atom_a in "
           "BOND::Memory_Allocate");
  if (!Malloc_Safely((void **)&(this->h_atom_b),
                     sizeof(int) * this->bond_numbers))
    printf("        Error occurs when malloc BOND::h_atom_b in "
           "BOND::Memory_Allocate");
  if (!Malloc_Safely((void **)&(this->h_k), sizeof(float) * this->bond_numbers))
    printf(
        "        Error occurs when malloc BOND::h_k in BOND::Memory_Allocate");
  if (!Malloc_Safely((void **)&(this->h_r0),
                     sizeof(float) * this->bond_numbers))
    printf(
        "        Error occurs when malloc BOND::h_r0 in BOND::Memory_Allocate");
  if (!Malloc_Safely((void **)&(this->h_bond_ene),
                     sizeof(float) * this->bond_numbers))
    printf("        Error occurs when malloc BOND::h_bond_ene in "
           "BOND::Memory_Allocate");
  if (!Malloc_Safely((void **)&(this->h_sigma_of_bond_ene), sizeof(float)))
    printf("        Error occurs when malloc BOND::h_sigma_of_bond_ene in "
           "BOND::Memory_Allocate");

  if (!Cuda_Malloc_Safely((void **)&this->d_atom_a,
                          sizeof(int) * this->bond_numbers))
    printf("        Error occurs when CUDA malloc BOND::d_atom_a in "
           "BOND::Memory_Allocate");
  if (!Cuda_Malloc_Safely((void **)&this->d_atom_b,
                          sizeof(int) * this->bond_numbers))
    printf("        Error occurs when CUDA malloc BOND::d_atom_b in "
           "BOND::Memory_Allocate");
  if (!Cuda_Malloc_Safely((void **)&this->d_k,
                          sizeof(float) * this->bond_numbers))
    printf("        Error occurs when CUDA malloc BOND::d_k in "
           "BOND::Memory_Allocate");
  if (!Cuda_Malloc_Safely((void **)&this->d_r0,
                          sizeof(float) * this->bond_numbers))
    printf("        Error occurs when CUDA malloc BOND::d_r0 in "
           "BOND::Memory_Allocate");
  if (!Cuda_Malloc_Safely((void **)&this->d_bond_ene,
                          sizeof(float) * this->bond_numbers))
    printf("        Error occurs when CUDA malloc BOND::d_bond_ene in "
           "BOND::Memory_Allocate");
  if (!Cuda_Malloc_Safely((void **)&this->d_sigma_of_bond_ene, sizeof(float)))
    printf("        Error occurs when CUDA malloc BOND::d_sigma_of_bond_ene in "
           "BOND::Memory_Allocate");
}

void BOND::Parameter_Host_To_Device() {
  cudaMemcpy(this->d_atom_a, this->h_atom_a, sizeof(int) * this->bond_numbers,
             cudaMemcpyHostToDevice);
  cudaMemcpy(this->d_atom_b, this->h_atom_b, sizeof(int) * this->bond_numbers,
             cudaMemcpyHostToDevice);
  cudaMemcpy(this->d_k, this->h_k, sizeof(float) * this->bond_numbers,
             cudaMemcpyHostToDevice);
  cudaMemcpy(this->d_r0, this->h_r0, sizeof(float) * this->bond_numbers,
             cudaMemcpyHostToDevice);
  cudaMemcpy(this->d_bond_ene, this->h_bond_ene,
             sizeof(float) * this->bond_numbers, cudaMemcpyHostToDevice);
  cudaMemcpy(this->d_sigma_of_bond_ene, this->h_sigma_of_bond_ene,
             sizeof(float), cudaMemcpyHostToDevice);
}

void BOND::Clear() {
  if (is_initialized) {
    cudaFree(this->d_atom_a);
    cudaFree(this->d_atom_b);
    cudaFree(this->d_k);
    cudaFree(this->d_r0);
    cudaFree(this->d_bond_ene);
    cudaFree(this->d_sigma_of_bond_ene);

    free(this->h_atom_a);
    free(this->h_atom_b);
    free(this->h_k);
    free(this->h_r0);
    free(this->h_bond_ene);
    free(this->h_sigma_of_bond_ene);

    h_atom_a = NULL;
    d_atom_a = NULL;
    h_atom_b = NULL;
    d_atom_b = NULL;
    d_k = NULL;
    h_k = NULL;
    d_r0 = NULL;
    h_r0 = NULL;

    h_bond_ene = NULL;
    d_bond_ene = NULL;
    d_sigma_of_bond_ene = NULL;
    h_sigma_of_bond_ene = NULL;

    is_initialized = 0;
  }
}

float BOND::Get_Energy(const UNSIGNED_INT_VECTOR *uint_crd, const VECTOR scaler,
                       int is_download) {
  if (is_initialized) {
    Bond_Energy_CUDA<<<(unsigned int)ceilf((float)this->bond_numbers /
                                           this->threads_per_block),
                       this->threads_per_block>>>(
        this->bond_numbers, uint_crd, scaler, this->d_atom_a, this->d_atom_b,
        this->d_k, this->d_r0, this->d_bond_ene);

    Sum_Of_List<<<1, 1024>>>(this->bond_numbers, this->d_bond_ene,
                             this->d_sigma_of_bond_ene);
    if (is_download) {
      cudaMemcpy(this->h_sigma_of_bond_ene, this->d_sigma_of_bond_ene,
                 sizeof(float), cudaMemcpyDeviceToHost);
      return h_sigma_of_bond_ene[0];
    } else {
      return 0;
    }
  }
  return NAN;
}

void BOND::Bond_Force_With_Atom_Energy_And_Virial(
    const UNSIGNED_INT_VECTOR *uint_crd, const VECTOR scaler, VECTOR *frc,
    float *atom_energy, float *atom_virial) {
  if (is_initialized) {
    Bond_Force_With_Atom_Energy_And_Virial_CUDA<<<
        (unsigned int)ceilf((float)this->bond_numbers /
                            this->threads_per_block),
        this->threads_per_block>>>(this->bond_numbers, uint_crd, scaler,
                                   this->d_atom_a, this->d_atom_b, this->d_k,
                                   this->d_r0, frc, atom_energy, atom_virial);
  }
}
