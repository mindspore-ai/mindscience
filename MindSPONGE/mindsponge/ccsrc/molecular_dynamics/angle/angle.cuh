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

//更加详细的类似备注请见bond模块

#ifndef ANGLE_CUH
#define ANGLE_CUH
#include "../common.cuh"
#include "../control.cuh"

struct ANGLE {
  char module_name[CHAR_LENGTH_MAX];
  int is_initialized = 0;
  int is_controller_printf_initialized = 0;
  int last_modify_date = 20210830;

  // E_angle = k (∠abc - theta0) ^ 2
  int angle_numbers = 0;
  int *h_atom_a = NULL;
  int *h_atom_b = NULL;
  int *h_atom_c = NULL;
  int *d_atom_a = NULL;
  int *d_atom_b = NULL;
  int *d_atom_c = NULL;
  float *h_angle_k = NULL;
  float *d_angle_k = NULL;
  float *h_angle_theta0 = NULL;
  float *d_angle_theta0 = NULL;

  float *h_angle_ene = NULL;
  float *d_angle_ene = NULL;
  float *d_sigma_of_angle_ene = NULL;
  float *h_sigma_of_angle_ene = NULL;

  // cuda计算分配相关参数
  int threads_per_block = 128;

  //初始化模块
  void Initial(CONTROLLER *controller, const char *module_name = NULL);
  //清空模块
  void Clear();
  //内存分配
  void Memory_Allocate();
  //从parm7文件中读取角信息
  void Read_Information_From_AMBERFILE(const char *file_name,
                                       CONTROLLER controller);
  //拷贝cpu中的数据到gpu中
  void Parameter_Host_To_Device();

  //计算angle force并同时计算能量并将其加到原子能量列表上
  void Angle_Force_With_Atom_Energy(const UNSIGNED_INT_VECTOR *uint_crd,
                                    const VECTOR scaler, VECTOR *frc,
                                    float *atom_energy);
  //获得能量
  float Get_Energy(const UNSIGNED_INT_VECTOR *unit_crd, const VECTOR scaler,
                   int is_download = 1);
};

#endif // ANGLE(angle.cuh)
