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

#include "cmap.cuh"

//由于求导带来的系数矩阵的逆矩阵A_inv
static const float A_inv[16][16] = {
    {
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    },
    {
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    },
    {
        -3,
        0,
        3,
        0,
        0,
        0,
        0,
        0,
        -2,
        0,
        -1,
        0,
        0,
        0,
        0,
        0,
    },
    {
        2,
        0,
        -2,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
    },
    {
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    },
    {
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
    },
    {
        0,
        0,
        0,
        0,
        -3,
        0,
        3,
        0,
        0,
        0,
        0,
        0,
        -2,
        0,
        -1,
        0,
    },
    {
        0,
        0,
        0,
        0,
        2,
        0,
        -2,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        1,
        0,
    },
    {
        -3,
        3,
        0,
        0,
        -2,
        -1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    },
    {
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        -3,
        3,
        0,
        0,
        -2,
        -1,
        0,
        0,
    },
    {
        9,
        -9,
        -9,
        9,
        6,
        3,
        -6,
        -3,
        6,
        -6,
        3,
        -3,
        4,
        2,
        2,
        1,
    },
    {
        -6,
        6,
        6,
        -6,
        -4,
        -2,
        4,
        2,
        -3,
        3,
        -3,
        3,
        -2,
        -1,
        -2,
        -1,
    },
    {
        2,
        -2,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    },
    {
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        2,
        -2,
        0,
        0,
        1,
        1,
        0,
        0,
    },
    {
        -6,
        6,
        6,
        -6,
        -3,
        -3,
        3,
        3,
        -4,
        4,
        -2,
        2,
        -2,
        -2,
        -1,
        -1,
    },
    {4, -4, -4, 4, 2, 2, -2, -2, 2, -2, 2, -2, 1, 1, 1, 1}};

void CMAP::Initial(CONTROLLER *controller, const char *module_name) {
  controller[0].printf("START INITIALIZING CMAP:\n");
  if (module_name == NULL) {
    strcpy(this->module_name, "cmap");
  } else {
    strcpy(this->module_name, module_name);
  }

  if (controller[0].Command_Exist(this->module_name, "in_file")) {
    FILE *fp = NULL;
    Open_File_Safely(&fp, controller[0].Command(this->module_name, "in_file"),
                     "r");

    int ret = fscanf(fp, "%d", &(this->tot_cmap_num));
    ret = fscanf(fp, "%d", &(this->uniq_cmap_num));
    controller->printf(
        "    total CMAP number is %d\n    unique CMAP number is %d\n",
        this->tot_cmap_num, this->uniq_cmap_num);
    this->Memory_Allocate();
    for (int i = 0; i < (this->uniq_cmap_num); i++) {
      ret = fscanf(fp, "%d", &cmap_resolution[i]);
      uniq_gridpoint_num += cmap_resolution[i] * cmap_resolution[i];
    }

    Malloc_Safely((void **)&(this->grid_value),
                  sizeof(float) * (this->uniq_cmap_num) * pow(24, 2));

    int temp = 0;
    for (int count = 0; count < this->uniq_cmap_num; count++) {
      for (int i = 0; i < pow(this->cmap_resolution[count], 2); i++) {
        ret = fscanf(fp, "%f", &grid_value[i + temp]);
      }
      temp += pow(this->cmap_resolution[count], 2);
    }

    for (int i = 0; i < (this->tot_cmap_num); i++) {
      //数组原子编号从0记
      ret = fscanf(fp, "%d", &this->h_atom_a[i]);
      ret = fscanf(fp, "%d", &this->h_atom_b[i]);
      ret = fscanf(fp, "%d", &this->h_atom_c[i]);
      ret = fscanf(fp, "%d", &this->h_atom_d[i]);
      ret = fscanf(fp, "%d", &this->h_atom_e[i]);
      ret = fscanf(fp, "%d", &this->cmap_type[i]);
    }

    for (int i = 0; i < (this->tot_cmap_num); i++) {
      tot_gridpoint_num +=
          cmap_resolution[cmap_type[i]] * cmap_resolution[cmap_type[i]];
    }

    Malloc_Safely((void **)&(this->inter_coeff),
                  sizeof(float) * 16 * tot_gridpoint_num);
    Cuda_Malloc_Safely((void **)&(this->d_inter_coeff),
                       sizeof(float) * 16 * tot_gridpoint_num);

    fclose(fp);
    is_initialized = 1;
  } else if (controller[0].Command_Exist("amber_parm7")) {
    Read_Information_From_AMBERFILE(controller[0].Command("amber_parm7"),
                                    controller[0]);
  }

  if (is_initialized && !is_controller_printf_initialized) {
    controller[0].Step_Print_Initial(this->module_name, "%.6f");
    is_controller_printf_initialized = 1;
    controller[0].printf("    structure last modify date is %d\n",
                         last_modify_date);
  }
  if (is_initialized) {
    //完成插值系数计算，完成初始化
    this->Interpolation(this->cmap_resolution, this->grid_value, controller[0]);

    Parameter_Host_to_Device();
    controller[0].printf("END INITIALIZING CMAP\n\n");
  } else {
    controller[0].printf("CMAP IS NOT INITIALIZED\n\n");
  }
}

void CMAP::Read_Information_From_AMBERFILE(const char *file_name,
                                           CONTROLLER controller) {
  //参数中的双二面角的信息

  FILE *parm = NULL;
  Open_File_Safely(&parm, file_name, "r");

  controller.printf("    Reading CAMP information from AMBER file:\n");

  char temps[CHAR_LENGTH_MAX];
  char temp_first_str[CHAR_LENGTH_MAX];
  char temp_second_str[CHAR_LENGTH_MAX];

  //中间/循环变量
  int count = 0, temp = 0;

  while (true) {
    if (fgets(temps, CHAR_LENGTH_MAX, parm) == NULL) {
      break;
    }
    if (sscanf(temps, "%s %s", temp_first_str, temp_second_str) != 2) {
      continue;
    }

    if (strcmp(temp_first_str, "%FLAG") == 0 &&
            strcmp(temp_second_str, "CMAP_COUNT") == 0 ||
        strcmp(temp_second_str, "CHARMM_CMAP_COUNT") == 0) {
      //读取parm7中的"COMMENT ..."(如果存在)以及"%FORMAT(2I8)" 两行
      char *get_value = fgets(temps, CHAR_LENGTH_MAX, parm);
      if (strncmp(temps, "%COMMENT", 8) == 0)
        get_value = fgets(temps, CHAR_LENGTH_MAX, parm);

      //读取CMAP个数
      int ret = fscanf(parm, "%d", &(this->tot_cmap_num));
      ret = fscanf(parm, "%d", &(this->uniq_cmap_num));

      controller.printf(
          "        total CMAP number is %d\n        unique CMAP number is %d\n",
          this->tot_cmap_num, this->uniq_cmap_num);
      this->Memory_Allocate();
    }

    if (strcmp(temp_first_str, "%FLAG") == 0 &&
            strcmp(temp_second_str, "CMAP_RESOLUTION") == 0 ||
        strcmp(temp_second_str, "CHARMM_CMAP_RESOLUTION") == 0) {
      //读取到"%FORMAT(20I4)"一行
      char *ret = fgets(temps, CHAR_LENGTH_MAX, parm);
      if (strncmp(temps, "%COMMENT", 8) == 0)
        ret = fgets(temps, CHAR_LENGTH_MAX, parm);

      for (int i = 0; i < (this->uniq_cmap_num); i++) {
        int ret2 = fscanf(parm, "%d", &cmap_resolution[i]);
        uniq_gridpoint_num += cmap_resolution[i] * cmap_resolution[i];
      }
      //读入全部双二面角信息并选择使用到的进行插值
      if (!Malloc_Safely((void **)&(this->grid_value),
                         sizeof(float) * (this->uniq_cmap_num) * pow(24, 2))) {
        printf("        Error occurs when malloc CMAP grid values in "
               "CMAP::Read_Information_From_AMBERFILE");
      }
    }

    //循环读取插值格点处的值，并将插值得到的系数保存
    if (strcmp(temp_first_str, "%FLAG") == 0 &&
        (strncmp(temp_second_str, "CMAP_PARAMETER", 14) == 0 ||
         strncmp(temp_second_str, "CHARMM_CMAP_PARAMETER", 15) == 0)) {
      char *ret = fgets(temps, CHAR_LENGTH_MAX, parm);
      if (strncmp(temps, "%COMMENT", 8) == 0)
        ret = fgets(temps, CHAR_LENGTH_MAX, parm);

      ////将所有格点值读取到一个数组中
      for (int i = 0; i < pow(this->cmap_resolution[count], 2); i++) {
        int ret2 = fscanf(parm, "%f", &grid_value[i + temp]);
      }

      temp += pow(this->cmap_resolution[count], 2);
      count += 1;
    }

    //读取参与双二面角的原子编号
    if (strcmp(temp_first_str, "%FLAG") == 0 &&
            strcmp(temp_second_str, "CMAP_INDEX") == 0 ||
        strcmp(temp_second_str, "CHARMM_CMAP_INDEX") == 0) {
      char *ret = fgets(temps, CHAR_LENGTH_MAX, parm);
      if (strncmp(temps, "%COMMENT", 8) == 0)
        ret = fgets(temps, CHAR_LENGTH_MAX, parm);

      for (int i = 0; i < (this->tot_cmap_num); i++) {
        //数组原子编号从0记
        int ret2 = fscanf(parm, "%d", &this->h_atom_a[i]);
        h_atom_a[i] -= 1;
        ret2 = fscanf(parm, "%d", &this->h_atom_b[i]);
        h_atom_b[i] -= 1;
        ret2 = fscanf(parm, "%d", &this->h_atom_c[i]);
        h_atom_c[i] -= 1;
        ret2 = fscanf(parm, "%d", &this->h_atom_d[i]);
        h_atom_d[i] -= 1;
        ret2 = fscanf(parm, "%d", &this->h_atom_e[i]);
        h_atom_e[i] -= 1;
        ret2 = fscanf(parm, "%d", &this->cmap_type[i]);
        cmap_type[i] -= 1;
      }
    }
  }

  for (int i = 0; i < (this->tot_cmap_num); i++) {
    tot_gridpoint_num +=
        cmap_resolution[cmap_type[i]] * cmap_resolution[cmap_type[i]];
  }

  if (!Malloc_Safely((void **)&(this->inter_coeff),
                     sizeof(float) * 16 * tot_gridpoint_num)) {
    printf("        Error occurs when malloc CMAP coefficients in "
           "CMAP::Read_Information_From_AMBERFILE");
  }
  if (!Cuda_Malloc_Safely((void **)&(this->d_inter_coeff),
                          sizeof(float) * 16 * tot_gridpoint_num)) {
    printf("        Error occurs when CUDA malloc CMAP coefficients in "
           "CMAP::Read_Information_From_AMBERFILE");
  }

  count = 0;
  temp = 0;
  is_initialized = 1;

  fclose(parm);

  if (this->tot_cmap_num == 0) {
    Clear();
  }
}

void CMAP::Parameter_Host_to_Device() {
  //原子序号
  cudaMemcpy(this->d_atom_a, this->h_atom_a, sizeof(int) * this->tot_cmap_num,
             cudaMemcpyHostToDevice);
  cudaMemcpy(this->d_atom_b, this->h_atom_b, sizeof(int) * this->tot_cmap_num,
             cudaMemcpyHostToDevice);
  cudaMemcpy(this->d_atom_c, this->h_atom_c, sizeof(int) * this->tot_cmap_num,
             cudaMemcpyHostToDevice);
  cudaMemcpy(this->d_atom_d, this->h_atom_d, sizeof(int) * this->tot_cmap_num,
             cudaMemcpyHostToDevice);
  cudaMemcpy(this->d_atom_e, this->h_atom_e, sizeof(int) * this->tot_cmap_num,
             cudaMemcpyHostToDevice);

  cudaMemcpy(this->d_cmap_resolution, this->cmap_resolution,
             sizeof(int) * (this->uniq_cmap_num), cudaMemcpyHostToDevice);
  cudaMemcpy(this->d_cmap_type, this->cmap_type,
             sizeof(int) * (this->tot_cmap_num), cudaMemcpyHostToDevice);
  //插值矩阵
  cudaMemcpy(this->d_inter_coeff, this->inter_coeff,
             sizeof(float) * (this->tot_cmap_num) * 24 * 24 * 16,
             cudaMemcpyHostToDevice);
}

void CMAP::Memory_Allocate() {
  // cmap相关信息
  if (!Malloc_Safely((void **)&(this->cmap_resolution),
                     sizeof(int) * (this->uniq_cmap_num))) {
    printf("        Error occurs when malloc cmap resolution in "
           "CMAP::Read_Information_From_AMBERFILE");
  }
  if (!Cuda_Malloc_Safely((void **)&(this->d_cmap_resolution),
                          sizeof(int) * (this->uniq_cmap_num))) {
    printf("        Error occurs when CUDA malloc cmap resolution in "
           "CMAP::Read_Information_From_AMBERFILE");
  }
  if (!Malloc_Safely((void **)&(this->cmap_type),
                     sizeof(int) * (this->tot_cmap_num))) {
    printf("        Error occurs when malloc cmap type in "
           "CMAP::Read_Information_From_AMBERFILE");
  }
  if (!Cuda_Malloc_Safely((void **)&(this->d_cmap_type),
                          sizeof(int) * (this->tot_cmap_num))) {
    printf("        Error occurs when CUDA malloc cmap type in "
           "CMAP::Read_Information_From_AMBERFILE");
  }

  //关于能量信息
  if (!Malloc_Safely((void **)&(this->h_sigma_of_cmap_ene), sizeof(float))) {
    printf("        Error occurs when malloc CMAP sum of energy in "
           "CMAP::Read_Information_From_AMBERFILE");
  }
  if (!Cuda_Malloc_Safely((void **)&(this->d_cmap_ene),
                          sizeof(float) * (this->tot_cmap_num))) {
    printf("        Error occurs when CUDA malloc CMAP energy in "
           "CMAP::Read_Information_From_AMBERFILE");
  }
  if (!Cuda_Malloc_Safely((void **)&(this->d_sigma_of_cmap_ene),
                          sizeof(float))) {
    printf("        Error occurs when CUDA malloc CMAP sum of energy in "
           "CMAP::Read_Information_From_AMBERFILE");
  }

  // cmap_test_temp
  if (!Malloc_Safely((void **)&(this->h_cmap_force), sizeof(float) * 45)) {
    printf("        Error occurs when malloc CMAP sum of energy in "
           "CMAP::Read_Information_From_AMBERFILE");
  }
  if (!Cuda_Malloc_Safely((void **)&(this->d_cmap_force), sizeof(float) * 45)) {
    printf("        Error occurs when CUDA malloc CMAP energy in "
           "CMAP::Read_Information_From_AMBERFILE");
  }

  //关于原子坐标
  if (!Malloc_Safely((void **)&this->h_atom_a,
                     sizeof(int) * this->tot_cmap_num))
    printf(
        "Error occurs when malloc DIHEDARL::h_atom_a in CMAP::cmap_initialize");
  if (!Malloc_Safely((void **)&this->h_atom_b,
                     sizeof(int) * this->tot_cmap_num))
    printf(
        "Error occurs when malloc DIHEDARL::h_atom_b in CMAP::cmap_initialize");
  if (!Malloc_Safely((void **)&this->h_atom_c,
                     sizeof(int) * this->tot_cmap_num))
    printf(
        "Error occurs when malloc DIHEDARL::h_atom_c in CMAP::cmap_initialize");
  if (!Malloc_Safely((void **)&this->h_atom_d,
                     sizeof(int) * this->tot_cmap_num))
    printf(
        "Error occurs when malloc DIHEDARL::h_atom_d in CMAP::cmap_initialize");
  if (!Malloc_Safely((void **)&this->h_atom_e,
                     sizeof(int) * this->tot_cmap_num))
    printf(
        "Error occurs when malloc DIHEDARL::h_atom_e in CMAP::cmap_initialize");

  if (!Cuda_Malloc_Safely((void **)&this->d_atom_a,
                          sizeof(int) * this->tot_cmap_num))
    printf("Error occurs when CUDA malloc CMAP::d_atom_a in "
           "CMAP::Dihedral_Initialize");
  if (!Cuda_Malloc_Safely((void **)&this->d_atom_b,
                          sizeof(int) * this->tot_cmap_num))
    printf("Error occurs when CUDA malloc CMAP::d_atom_b in "
           "CMAP::Dihedral_Initialize");
  if (!Cuda_Malloc_Safely((void **)&this->d_atom_c,
                          sizeof(int) * this->tot_cmap_num))
    printf("Error occurs when CUDA malloc CMAP::d_atom_c in "
           "CMAP::Dihedral_Initialize");
  if (!Cuda_Malloc_Safely((void **)&this->d_atom_d,
                          sizeof(int) * this->tot_cmap_num))
    printf("Error occurs when CUDA malloc CMAP::d_atom_d in "
           "CMAP::Dihedral_Initialize");
  if (!Cuda_Malloc_Safely((void **)&this->d_atom_e,
                          sizeof(int) * this->tot_cmap_num))
    printf("Error occurs when CUDA malloc CMAP::d_atom_e in "
           "CMAP::Dihedral_Initialize");
}

void CMAP::Clear() {
  if (is_initialized) {
    is_initialized = 0;
    is_controller_printf_initialized = 0;
    tot_cmap_num = 0;
    uniq_cmap_num = 0;
    tot_gridpoint_num = 0;
    uniq_gridpoint_num = 0;

    free(this->h_atom_a);
    free(this->h_atom_b);
    free(this->h_atom_c);
    free(this->h_atom_d);
    free(this->h_atom_e);
    free(this->cmap_resolution);
    free(this->cmap_type);
    free(this->grid_value);
    free(this->inter_coeff);
    free(this->h_sigma_of_cmap_ene);
    free(this->h_cmap_force); // cmap_test_temp

    cudaFree(this->d_atom_a);
    cudaFree(this->d_atom_b);
    cudaFree(this->d_atom_c);
    cudaFree(this->d_atom_d);
    cudaFree(this->d_atom_e);
    cudaFree(this->d_cmap_resolution);
    cudaFree(this->d_inter_coeff);
    cudaFree(this->d_cmap_type);
    cudaFree(this->d_cmap_ene);
    cudaFree(this->d_sigma_of_cmap_ene);
    cudaFree(this->d_cmap_force); // cmap_test_temp

    this->h_atom_a = NULL;
    this->h_atom_b = NULL;
    this->h_atom_c = NULL;
    this->h_atom_d = NULL;
    this->h_atom_e = NULL;

    this->cmap_resolution = NULL;
    this->cmap_type = NULL;
    this->grid_value = NULL;
    this->inter_coeff = NULL;
    this->h_sigma_of_cmap_ene = NULL;

    this->d_atom_a = NULL;
    this->d_atom_b = NULL;
    this->d_atom_c = NULL;
    this->d_atom_d = NULL;
    this->d_atom_e = NULL;
    this->d_cmap_resolution = NULL;
    this->d_inter_coeff = NULL;
    this->d_cmap_ene = NULL;
    this->d_sigma_of_cmap_ene = NULL;
  }
}

void CMAP::Interpolation(int *resolution, float *grid_value,
                         CONTROLLER controller) {
  //临时储存节点的值和差分
  float f[4][4];
  float p[16];

  printf("    Start Interpolating the CMAP Grid Value\n");
  //首先从统一读入的CMAP格点数据中截取出需要插值的数据
  int temp_loca = 0, temp_record = 0; //记录位置
  float *temp_grid_value;             //临时储存特定CMAP格点的值
  float *temp_inter_coeff;            //临时储存插值系数
  int temp_type;                      //标记CMAP类型
  int temp_reso;                      //标记格点分辨率

  int phi_index = 0, psi_index = 0;
  //插值数据结构为：
  //                          psi
  //                - - - - - ... - - - - -
  //                - - - - - ... - - - - -
  //           phi        .
  //                      .
  //                                       .
  //                - - - - - ... - - - - -
  //规模为 resolution*resolution
  for (int k = 0; k < (this->tot_cmap_num); k++) {
    temp_type = this->cmap_type[k];
    temp_reso = this->cmap_resolution[temp_type];
    // controller.printf("        CMAP type number is %d, CMAP resolution is
    // %d\n", this->cmap_type[k], temp_reso);

    //临时数组，用于储存单个CMAP的插值数据与计算得到的插值系数
    if (!Malloc_Safely((void **)&(temp_grid_value),
                       sizeof(float) * pow(temp_reso, 2))) {
      printf("        Error occurs when malloc temprerary grid value list in "
             "CMAP::Interpolation");
    }
    if (!Malloc_Safely((void **)&(temp_inter_coeff),
                       sizeof(float) * 16 * pow(temp_reso, 2))) {
      printf("        Error occurs when malloc temprerary interpolation "
             "coefficients list in CMAP::Interpolation");
    }

    // temp_loca = pow(this->cmap_resolution[0], 2)*(temp_type);
    // //demo，在所有双二面角插值分辨率相同时是正确的
    temp_loca = 0;
    for (int i = 0; i < temp_type; i++) {
      temp_loca += (this->cmap_resolution[i]) * (this->cmap_resolution[i]);
    }
    memcpy(temp_grid_value, &(this->grid_value[temp_loca]),
           sizeof(float) * pow(temp_reso, 2));

    for (int i = 0; i < (temp_reso) * (temp_reso); i++) {
      //对每个单元进行插值
      psi_index = i % (temp_reso);
      phi_index = (i - psi_index) / (temp_reso);
      for (int m = 0; m < 4; m++) {
        for (int n = 0; n < 4; n++) {
          //引入周期性的读取方式
          if (phi_index + m - 1 >= 0 && psi_index + n - 1 >= 0)
            f[m][n] = temp_grid_value[((phi_index + m - 1) % (temp_reso)) *
                                          temp_reso +
                                      (psi_index + n - 1) % temp_reso];
          else if ((phi_index + m - 1 < 0 && psi_index + n - 1 >= 0))
            f[m][n] = temp_grid_value[((phi_index + m + 23) % (temp_reso)) *
                                          temp_reso +
                                      (psi_index + n - 1) % temp_reso];
          else if ((phi_index + m - 1 >= 0 && psi_index + n - 1 < 0))
            f[m][n] = temp_grid_value[((phi_index + m - 1) % (temp_reso)) *
                                          temp_reso +
                                      (psi_index + n + 23) % temp_reso];
          else
            f[m][n] = temp_grid_value[((phi_index + m + 23) % (temp_reso)) *
                                          temp_reso +
                                      (psi_index + n + 23) % temp_reso];
        }
      }
      //格点值以及一阶二阶差分
      p[0] = f[1][1];
      p[1] = f[2][1];
      p[2] = f[1][2];
      p[3] = f[2][2];
      p[4] = (f[2][1] - f[0][1]) / 2;
      p[5] = (f[3][1] - f[1][1]) / 2;
      p[6] = (f[2][2] - f[0][2]) / 2;
      p[7] = (f[3][2] - f[1][2]) / 2;
      p[8] = (f[1][2] - f[1][0]) / 2;
      p[9] = (f[2][2] - f[2][0]) / 2;
      p[10] = (f[1][3] - f[1][1]) / 2;
      p[11] = (f[2][3] - f[2][1]) / 2;
      p[12] = (f[2][2] + f[0][0] - f[2][0] - f[0][2]) / 4;
      p[13] = (f[3][2] + f[1][0] - f[3][0] - f[1][2]) / 4;
      p[14] = (f[2][3] + f[0][1] - f[2][1] - f[0][3]) / 4;
      p[15] = (f[3][3] + f[1][1] - f[3][1] - f[1][3]) / 4;

      //系数矩阵（size:4*4）的对应关系为列指标对应y次数，行指标对应x次数，原始数据（size:reso*reso）行指标对应x坐标，列指标对应y坐标
      for (int q = 0; q < 16; q++) {
        //手动矩阵乘法
        temp_inter_coeff[i * 16 + q] = 0;
        for (int j = 0; j < 16; j++)
          temp_inter_coeff[i * 16 + q] += (A_inv[q][j]) * p[j];
      }
    }

    //将插值系数存出来
    memcpy(&inter_coeff[temp_record], temp_inter_coeff,
           sizeof(float) * 16 * pow(temp_reso, 2));
    temp_record += temp_reso * temp_reso * 16;

    free(temp_inter_coeff);
    free(temp_grid_value);
  }
  temp_loca = 0;
  temp_record = 0;
  printf("    End Interpolating CMAP Grid Value\n");
}

static __global__ void CMAP_Force_with_Atom_Energy_CUDA(
    const int cmap_numbers, const UNSIGNED_INT_VECTOR *uint_crd,
    const VECTOR scaler, const int *atom_a, const int *atom_b,
    const int *atom_c, const int *atom_d, const int *atom_e,
    const int *resolution, const int *type, const float *inter_coeff,
    VECTOR *frc, float *ene) {
  int cmap_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (cmap_i < cmap_numbers) {
    int atom_i = atom_a[cmap_i];
    int atom_j = atom_b[cmap_i];
    int atom_k = atom_c[cmap_i];
    int atom_l = atom_d[cmap_i];
    int atom_m = atom_e[cmap_i];

    //计算phi
    VECTOR drij =
        Get_Periodic_Displacement(uint_crd[atom_i], uint_crd[atom_j], scaler);
    VECTOR drkj =
        Get_Periodic_Displacement(uint_crd[atom_k], uint_crd[atom_j], scaler);
    VECTOR drkl =
        Get_Periodic_Displacement(uint_crd[atom_k], uint_crd[atom_l], scaler);

    //法向量夹角
    VECTOR r1_phi = drij ^ drkj;
    VECTOR r2_phi = drkl ^ drkj;

    float r1_1_phi = rnorm3df(r1_phi.x, r1_phi.y, r1_phi.z);
    float r2_1_phi = rnorm3df(r2_phi.x, r2_phi.y, r2_phi.z);
    // float r1_2_phi = r1_1_phi * r1_1_phi;
    // float r2_2_phi = r2_1_phi * r2_1_phi;
    float r1_1_r2_1_phi = r1_1_phi * r2_1_phi;

    float phi = r1_phi * r2_phi * r1_1_r2_1_phi;
    phi = fmaxf(-0.999999, fminf(phi, 0.999999));
    phi = acosf(phi);

    // acosf()只能返回[0,pi],需要确定其正负，最终phi落在[-pi,pi]

    phi = CONSTANT_Pi - phi;

    float sign_phi = (r2_phi ^ r1_phi) * drkj;
    phi = copysignf(phi, sign_phi);

    float cos_phi = cosf(phi);
    float sin_phi = sinf(phi);

    //计算psi
    VECTOR drjk =
        Get_Periodic_Displacement(uint_crd[atom_j], uint_crd[atom_k], scaler);
    VECTOR drlk =
        Get_Periodic_Displacement(uint_crd[atom_l], uint_crd[atom_k], scaler);
    VECTOR drlm =
        Get_Periodic_Displacement(uint_crd[atom_l], uint_crd[atom_m], scaler);

    //法向量夹角
    VECTOR r1_psi = drjk ^ drlk;
    VECTOR r2_psi = drlm ^ drlk;

    float r1_1_psi = rnorm3df(r1_psi.x, r1_psi.y, r1_psi.z);
    float r2_1_psi = rnorm3df(r2_psi.x, r2_psi.y, r2_psi.z);
    // float r1_2_psi = r1_1_psi * r1_1_psi;
    // float r2_2_psi = r2_1_psi * r2_1_psi;
    float r1_1_r2_1_psi = r1_1_psi * r2_1_psi;

    float psi = r1_psi * r2_psi * r1_1_r2_1_psi;
    psi = fmaxf(-0.999999, fminf(psi, 0.999999));
    psi = acosf(psi);

    //同理将psi映射到[-pi,pi]
    psi = CONSTANT_Pi - psi;
    float sign_psi = (r2_psi ^ r1_psi) * drlk;
    psi = copysignf(psi, sign_psi);

    float cos_psi = cosf(psi);
    float sin_psi = sinf(psi);

    //计算能量
    //首先将phi,psi
    //对pi归一化,单位为(pi/resolution),并确定其所属格点以及在格内的位置

    phi = phi / (2.0 * CONSTANT_Pi / resolution[cmap_i]);
    psi = psi / (2.0 * CONSTANT_Pi / resolution[cmap_i]);

    float parm_phi = phi - floorf(phi);
    float parm_psi = psi - floorf(psi);
    int locate_phi = (int)floorf(phi) + 12;
    int locate_psi = (int)floorf(psi) + 12;

    //定义幂次
    float parm_phi_2 = parm_phi * parm_phi;
    float parm_phi_3 = parm_phi_2 * parm_phi;
    float parm_psi_2 = parm_psi * parm_psi;
    float parm_psi_3 = parm_psi_2 * parm_psi;

    //用于定位的中间变量
    // int locate = 16 * (locate_phi * 24 + locate_psi) + 24 * 24 * 16 * cmap_i;
    int temp_reso = resolution[type[cmap_i] - 1];
    int locate = 16 * (locate_phi * temp_reso + locate_psi);
    for (int i = 0; i < cmap_i; i++) {
      locate += 16 * resolution[type[i] - 1] * resolution[type[i] - 1];
    }

    //计算能量对有符号归一化二面角的偏微分
    float dE_dphi =
        (inter_coeff[locate + 4] + parm_psi * inter_coeff[locate + 5] +
         parm_psi_2 * inter_coeff[locate + 6] +
         parm_psi_3 * inter_coeff[locate + 7]) +
        2 * parm_phi *
            (inter_coeff[locate + 8] + parm_psi * inter_coeff[locate + 9] +
             parm_psi_2 * inter_coeff[locate + 10] +
             parm_psi_3 * inter_coeff[locate + 11]) +
        3 * parm_phi_2 *
            (inter_coeff[locate + 12] + parm_psi * inter_coeff[locate + 13] +
             parm_psi_2 * inter_coeff[locate + 14] +
             parm_psi_3 * inter_coeff[locate + 15]);

    float dE_dpsi =
        inter_coeff[locate + 1] + 2 * parm_psi * inter_coeff[locate + 2] +
        3 * parm_psi_2 * inter_coeff[locate + 3] +
        parm_phi *
            (inter_coeff[locate + 5] + 2 * parm_psi * inter_coeff[locate + 6] +
             3 * parm_psi_2 * inter_coeff[locate + 7]) +
        parm_phi_2 *
            (inter_coeff[locate + 9] + 2 * parm_psi * inter_coeff[locate + 10] +
             3 * parm_psi_2 * inter_coeff[locate + 11]) +
        parm_phi_3 * (inter_coeff[locate + 13] +
                      2 * parm_psi * inter_coeff[locate + 14] +
                      3 * parm_psi_2 * inter_coeff[locate + 15]);

    //将有符号归一化二面角映射回弧度制二面角
    dE_dphi = dE_dphi / (2.0 * CONSTANT_Pi / resolution[cmap_i]);
    dE_dpsi = dE_dpsi / (2.0 * CONSTANT_Pi / resolution[cmap_i]);

    // phi角部分
    VECTOR temp_phi_A = drij ^ drjk;
    VECTOR temp_phi_B = drlk ^ drjk;

    VECTOR dphi_dri =
        -sqrtf(drjk * drjk) / (temp_phi_A * temp_phi_A) * temp_phi_A;
    VECTOR dphi_drj =
        +sqrtf(drjk * drjk) / (temp_phi_A * temp_phi_A) * temp_phi_A +
        drij * drjk / (temp_phi_A * temp_phi_A * sqrtf(drjk * drjk)) *
            temp_phi_A -
        drlk * drjk / (temp_phi_B * temp_phi_B * sqrtf(drjk * drjk)) *
            temp_phi_B;
    VECTOR dphi_drk =
        -sqrtf(drjk * drjk) / (temp_phi_B * temp_phi_B) * temp_phi_B -
        drij * drjk / (temp_phi_A * temp_phi_A * sqrtf(drjk * drjk)) *
            temp_phi_A +
        drlk * drjk / (temp_phi_B * temp_phi_B * sqrtf(drjk * drjk)) *
            temp_phi_B;
    VECTOR dphi_drl =
        +sqrtf(drjk * drjk) / (temp_phi_B * temp_phi_B) * temp_phi_B;
    VECTOR dphi_drm = {0, 0, 0};

    // psi角部分
    VECTOR drml =
        Get_Periodic_Displacement(uint_crd[atom_m], uint_crd[atom_l], scaler);

    VECTOR temp_psi_A = drjk ^ drkl;
    VECTOR temp_psi_B = drml ^ drkl;

    VECTOR dpsi_dri = {0, 0, 0};
    VECTOR dpsi_drj =
        -sqrtf(drkl * drkl) / (temp_psi_A * temp_psi_A) * temp_psi_A;
    VECTOR dpsi_drk =
        sqrtf(drkl * drkl) / (temp_psi_A * temp_psi_A) * temp_psi_A +
        drjk * drkl / (temp_psi_A * temp_psi_A * sqrtf(drkl * drkl)) *
            temp_psi_A -
        drml * drkl / (temp_psi_B * temp_psi_B * sqrtf(drkl * drkl)) *
            temp_psi_B;
    VECTOR dpsi_drl =
        -sqrtf(drkl * drkl) / (temp_psi_B * temp_psi_B) * temp_psi_B -
        drjk * drkl / (temp_psi_A * temp_psi_A * sqrtf(drkl * drkl)) *
            temp_psi_A +
        drml * drkl / (temp_psi_B * temp_psi_B * sqrtf(drkl * drkl)) *
            temp_psi_B;
    VECTOR dpsi_drm =
        sqrtf(drkl * drkl) / (temp_psi_B * temp_psi_B) * temp_psi_B;

    //计算力
    VECTOR fi = -(dE_dphi * dphi_dri + dE_dpsi * dpsi_dri);
    VECTOR fj = -(dE_dphi * dphi_drj + dE_dpsi * dpsi_drj);
    VECTOR fk = -(dE_dphi * dphi_drk + dE_dpsi * dpsi_drk);
    VECTOR fl = -(dE_dphi * dphi_drl + dE_dpsi * dpsi_drl);
    VECTOR fm = -(dE_dphi * dphi_drm + dE_dpsi * dpsi_drm);

    atomicAdd(&frc[atom_i].x, fi.x);
    atomicAdd(&frc[atom_i].y, fi.y);
    atomicAdd(&frc[atom_i].z, fi.z);
    atomicAdd(&frc[atom_j].x, fj.x);
    atomicAdd(&frc[atom_j].y, fj.y);
    atomicAdd(&frc[atom_j].z, fj.z);
    atomicAdd(&frc[atom_k].x, fk.x);
    atomicAdd(&frc[atom_k].y, fk.y);
    atomicAdd(&frc[atom_k].z, fk.z);
    atomicAdd(&frc[atom_l].x, fl.x);
    atomicAdd(&frc[atom_l].y, fl.y);
    atomicAdd(&frc[atom_l].z, fl.z);
    atomicAdd(&frc[atom_m].x, fm.x);
    atomicAdd(&frc[atom_m].y, fm.y);
    atomicAdd(&frc[atom_m].z, fm.z);

    //[1,phi,phi^2,phi^3]multiply inter_coeff(4*4,row priority)multiply
    //[1,psi,psi^2,psi^3]T
    float Energy = inter_coeff[locate] + parm_psi * inter_coeff[locate + 1] +
                   parm_psi_2 * inter_coeff[locate + 2] +
                   parm_psi_3 * inter_coeff[locate + 3] +
                   parm_phi * (inter_coeff[locate + 4] +
                               parm_psi * inter_coeff[locate + 5] +
                               parm_psi_2 * inter_coeff[locate + 6] +
                               parm_psi_3 * inter_coeff[locate + 7]) +
                   parm_phi_2 * (inter_coeff[locate + 8] +
                                 parm_psi * inter_coeff[locate + 9] +
                                 parm_psi_2 * inter_coeff[locate + 10] +
                                 parm_psi_3 * inter_coeff[locate + 11]) +
                   parm_phi_3 * (inter_coeff[locate + 12] +
                                 parm_psi * inter_coeff[locate + 13] +
                                 parm_psi_2 * inter_coeff[locate + 14] +
                                 parm_psi_3 * inter_coeff[locate + 15]);

    atomicAdd(&ene[atom_i], Energy);
  }
}

static __global__ void CMAP_Energy_CUDA(const int cmap_numbers,
                                        const UNSIGNED_INT_VECTOR *uint_crd,
                                        const VECTOR scaler, const int *atom_a,
                                        const int *atom_b, const int *atom_c,
                                        const int *atom_d, const int *atom_e,
                                        const int *resolution, const int *type,
                                        const float *inter_coeff, float *ene) {
  int cmap_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (cmap_i < cmap_numbers) {
    int atom_i = atom_a[cmap_i];
    int atom_j = atom_b[cmap_i];
    int atom_k = atom_c[cmap_i];
    int atom_l = atom_d[cmap_i];
    int atom_m = atom_e[cmap_i];

    //计算phi
    VECTOR drij =
        Get_Periodic_Displacement(uint_crd[atom_i], uint_crd[atom_j], scaler);
    VECTOR drkj =
        Get_Periodic_Displacement(uint_crd[atom_k], uint_crd[atom_j], scaler);
    VECTOR drkl =
        Get_Periodic_Displacement(uint_crd[atom_k], uint_crd[atom_l], scaler);

    //法向量夹角
    VECTOR r1_phi = drij ^ drkj;
    VECTOR r2_phi = drkl ^ drkj;

    float r1_1_phi = rnorm3df(r1_phi.x, r1_phi.y, r1_phi.z);
    float r2_1_phi = rnorm3df(r2_phi.x, r2_phi.y, r2_phi.z);
    // float r1_2_phi = r1_1_phi * r1_1_phi;
    // float r2_2_phi = r2_1_phi * r2_1_phi;
    float r1_1_r2_1_phi = r1_1_phi * r2_1_phi;

    float phi = r1_phi * r2_phi * r1_1_r2_1_phi;
    phi = fmaxf(-0.999999, fminf(phi, 0.999999));
    phi = acosf(phi);

    // acosf()只能返回[0,pi],需要确定其正负，最终phi落在[-pi,pi]

    phi = CONSTANT_Pi - phi;

    float sign = (r2_phi ^ r1_phi) * drkj;
    phi = copysignf(phi, sign);

    float cos_phi = cosf(phi);
    float sin_phi = sinf(phi);

    //计算psi
    VECTOR drjk =
        Get_Periodic_Displacement(uint_crd[atom_j], uint_crd[atom_k], scaler);
    VECTOR drlk =
        Get_Periodic_Displacement(uint_crd[atom_l], uint_crd[atom_k], scaler);
    VECTOR drlm =
        Get_Periodic_Displacement(uint_crd[atom_l], uint_crd[atom_m], scaler);

    //法向量夹角
    VECTOR r1_psi = drjk ^ drlk;
    VECTOR r2_psi = drlm ^ drlk;

    float r1_1_psi = rnorm3df(r1_psi.x, r1_psi.y, r1_psi.z);
    float r2_1_psi = rnorm3df(r2_psi.x, r2_psi.y, r2_psi.z);
    // float r1_2_psi = r1_1_psi * r1_1_psi;
    // float r2_2_psi = r2_1_psi * r2_1_psi;
    float r1_1_r2_1_psi = r1_1_psi * r2_1_psi;

    float psi = r1_psi * r2_psi * r1_1_r2_1_psi;
    psi = fmaxf(-0.999999, fminf(psi, 0.999999));
    psi = acosf(psi);

    //同理将psi映射到[-pi,pi]
    psi = CONSTANT_Pi - psi;
    sign = (r2_psi ^ r1_psi) * drlk;
    psi = copysignf(psi, sign);

    float cos_psi = cosf(psi);
    float sin_psi = sinf(psi);

    //计算能量
    //首先将phi,psi
    //对pi归一化,单位为(2pi/resolution),并确定其所属格点以及在格内的位置

    int temp_reso = resolution[type[cmap_i - 1]];
    phi = phi / (2.0 * CONSTANT_Pi / temp_reso);
    psi = psi / (2.0 * CONSTANT_Pi / temp_reso);

    float parm_phi = phi - floorf(phi);
    float parm_psi = psi - floorf(psi);
    int locate_phi = (int)floorf(phi) + 12;
    int locate_psi = (int)floorf(psi) + 12;

    //定义幂次
    float parm_phi_2 = parm_phi * parm_phi;
    float parm_phi_3 = parm_phi_2 * parm_phi;

    float parm_psi_2 = parm_psi * parm_psi;
    float parm_psi_3 = parm_psi_2 * parm_psi;

    //用于定位的中间变量
    int locate = 16 * (locate_phi * temp_reso + locate_psi);
    for (int i = 0; i < cmap_i; i++) {
      locate += 16 * resolution[type[i] - 1] * resolution[type[i] - 1];
    }

    //[1,phi,phi^2,phi^3]multiply inter_coeff(4*4,row priority)multiply
    //[1,psi,psi^2,psi^3]T
    ene[cmap_i] = inter_coeff[locate] + parm_psi * inter_coeff[locate + 1] +
                  parm_psi_2 * inter_coeff[locate + 2] +
                  parm_psi_3 * inter_coeff[locate + 3] +
                  parm_phi * (inter_coeff[locate + 4] +
                              parm_psi * inter_coeff[locate + 5] +
                              parm_psi_2 * inter_coeff[locate + 6] +
                              parm_psi_3 * inter_coeff[locate + 7]) +
                  parm_phi_2 * (inter_coeff[locate + 8] +
                                parm_psi * inter_coeff[locate + 9] +
                                parm_psi_2 * inter_coeff[locate + 10] +
                                parm_psi_3 * inter_coeff[locate + 11]) +
                  parm_phi_3 * (inter_coeff[locate + 12] +
                                parm_psi * inter_coeff[locate + 13] +
                                parm_psi_2 * inter_coeff[locate + 14] +
                                parm_psi_3 * inter_coeff[locate + 15]);

    temp_reso = 0;
    locate = 0;
  }
}

float CMAP::Get_Energy(const UNSIGNED_INT_VECTOR *uint_crd, const VECTOR scaler,
                       int is_download) {
  if (is_initialized) {
    CMAP_Energy_CUDA<<<(unsigned int)ceilf((float)this->tot_cmap_num /
                                           this->threads_per_block),
                       this->threads_per_block>>>(
        this->tot_cmap_num, uint_crd, scaler, this->d_atom_a, this->d_atom_b,
        this->d_atom_c, this->d_atom_d, this->d_atom_e, this->d_cmap_resolution,
        this->d_cmap_type, this->d_inter_coeff, this->d_cmap_ene);
    Sum_Of_List<<<1, 1024>>>(this->tot_cmap_num, this->d_cmap_ene,
                             this->d_sigma_of_cmap_ene);

    if (is_download) {
      cudaMemcpy(this->h_sigma_of_cmap_ene, this->d_sigma_of_cmap_ene,
                 sizeof(float), cudaMemcpyDeviceToHost);
      // cudaDeviceReset();
      return h_sigma_of_cmap_ene[0];
    } else {
      return 0;
    }
  }
  return NAN;
}

void CMAP::CMAP_Force_with_Atom_Energy(const UNSIGNED_INT_VECTOR *uint_crd,
                                       const VECTOR scaler, VECTOR *frc,
                                       float *atom_energy) {
  if (is_initialized == 1) {
    CMAP_Force_with_Atom_Energy_CUDA<<<(unsigned int)ceilf(
                                           (float)this->tot_cmap_num /
                                           this->threads_per_block),
                                       this->threads_per_block>>>(
        this->tot_cmap_num, uint_crd, scaler, this->d_atom_a, this->d_atom_b,
        this->d_atom_c, this->d_atom_d, this->d_atom_e, this->d_cmap_resolution,
        this->d_cmap_type, this->d_inter_coeff, frc, atom_energy);
  }
}
