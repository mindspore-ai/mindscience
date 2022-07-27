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

#ifndef PME_H
#define PME_H

#include <cufft.h>
#include "../common.cuh"
#include "../control.cuh"

struct Particle_Mesh_Ewald {
  char module_name[CHAR_LENGTH_MAX];
  int is_initialized = 0;
  int is_controller_printf_initialized = 0;
  int last_modify_date = 20210831;

  // fft维度参数
  int fftx = -1;
  int ffty = -1;
  int fftz = -1;
  int PME_Nall = 0;
  int PME_Nin = 0;
  int PME_Nfft = 0;

  // cuda参数
  dim3 thread_PME = {8, 8};
  cufftHandle PME_plan_r2c;
  cufftHandle PME_plan_c2r;

  //初始化参数
  int atom_numbers = 0;

  //体积相关的物理参数
  VECTOR boxlength;
  float *PME_BC = NULL; // GPU上的BC数组
  float *PME_BC0 =
      NULL; // GPU上的BC0数组，也即BC数组在乘上盒子相关信息之前的数组，更新体积的时候用
  VECTOR PME_inverse_box_vector;

  //体积无关的物理参数
  UNSIGNED_INT_VECTOR *PME_kxyz = NULL;
  UNSIGNED_INT_VECTOR *PME_uxyz = NULL;
  VECTOR *PME_frxyz = NULL;
  float *PME_Q = NULL;
  float *PME_FBCFQ = NULL;
  cufftComplex *PME_FQ = NULL;
  int **PME_atom_near = NULL;

  //控制参数
  float beta;
  float cutoff = 10.0;
  float tolerance = 0.00001f;

  //非中性时的能量额外项处理
  float neutralizing_factor = 0; //系数
  float *charge_sum = NULL;      //电荷量

  //能量参数
  float *d_direct_atom_energy = NULL;     //每个原子的直接的能量数组
  float *d_correction_atom_energy = NULL; //每个原子的修正能量数组
  float *d_reciprocal_ene = NULL;
  float *d_self_ene = NULL;
  float *d_direct_ene = NULL;
  float *d_correction_ene = NULL;
  float *d_ee_ene = NULL;
  float reciprocal_ene = 0;
  float self_ene = 0;
  float direct_ene = 0;
  float correction_ene = 0;
  float ee_ene;

  enum PME_ENERGY_PART {
    TOTAL = 0,
    DIRECT = 1,
    RECIPROCAL = 2,
    CORRECTION = 3,
    SELF = 4,
  };

  //初始化PME系统（PME信息）
  void Initial(CONTROLLER *controller, int atom_numbers, VECTOR boxlength,
               float cutoff, const char *module_name = NULL);
  //清除内存
  void Clear();

  /*-----------------------------------------------------------------------------------------
  下面的函数是普通md的需求
  ------------------------------------------------------------------------------------------*/

  //计算exclude能量和能量，并加到每个原子上
  void PME_Excluded_Force_With_Atom_Energy(const UNSIGNED_INT_VECTOR *uint_crd,
                                           const VECTOR sacler,
                                           const float *charge,
                                           const int *excluded_list_start,
                                           const int *excluded_list,
                                           const int *excluded_atom_numbers,
                                           VECTOR *frc, float *atom_energy);
  //计算倒空间力，并计算自能和倒空间的能量，并结合其他部分计算出PME部分给出的总维里（需要先计算其他部分）
  void PME_Reciprocal_Force_With_Energy_And_Virial(
      const UNSIGNED_INT_VECTOR *uint_crd, const float *charge, VECTOR *force,
      int need_virial, int need_energy, float *d_virial, float *d_potential);

  // float Get_Energy(const UNSIGNED_INT_VECTOR *uint_crd, const float *charge,
  // const ATOM_GROUP *nl, const VECTOR scaler,
  // const int *excluded_list_start, const int *excluded_list, const int
  //*excluded_atom_numbers, int is_download = 1);

  float Get_Energy(const UNSIGNED_INT_VECTOR *uint_crd, const float *charge,
                   const ATOM_GROUP *nl, const VECTOR scaler,
                   const int *excluded_list_start, const int *excluded_list,
                   const int *excluded_atom_numbers, int which_part = 0,
                   int is_download = 1);

  void Update_Volume(VECTOR boxlength);
  void Update_Box_Length(VECTOR boxlength);
};

__global__ void PME_Atom_Near(const UNSIGNED_INT_VECTOR *uint_crd,
                              int **PME_atom_near, const int PME_Nin,
                              const float periodic_factor_inverse_x,
                              const float periodic_factor_inverse_y,
                              const float periodic_factor_inverse_z,
                              const int atom_numbers, const int fftx,
                              const int ffty, const int fftz,
                              const UNSIGNED_INT_VECTOR *PME_kxyz,
                              UNSIGNED_INT_VECTOR *PME_uxyz, VECTOR *PME_frxyz);

__global__ void PME_Q_Spread(int **PME_atom_near, const float *charge,
                             const VECTOR *PME_frxyz, float *PME_Q,
                             const UNSIGNED_INT_VECTOR *PME_kxyz,
                             const int atom_numbers);

__global__ void PME_Energy_Product(const int element_number, const float *list1,
                                   const float *list2, float *sum);

__global__ void PME_BCFQ(cufftComplex *PME_FQ, float *PME_BC, int PME_Nfft);

#endif
