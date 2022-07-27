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

#ifndef TI_CORE_CUH
#define TI_CORE_CUH
#include "../PME_force/PME_force.cuh"
#include "../common.cuh"
#include "../control.cuh"

//普通分子模拟所涉及的大部分信息
struct TI_CORE {
  int is_initialized = 0;
  int last_modify_date = 20210525;

  // sponge输入初始化
  void Initial(CONTROLLER *controller);

  int atom_numbers = 0; //模拟的总原子数目
  VECTOR box_length;
  VECTOR last_box_length;
  float volume_change_factor;
  VECTOR box_angle;
  //每个原子的基本物理测量量，on host
  // VECTOR *velocity = NULL;
  VECTOR *coordinate = NULL;

  //每个原子的基本物理测量量，on device
  // VECTOR *vel = NULL;
  VECTOR *crd = NULL;
  UNSIGNED_INT_VECTOR *uint_crd = NULL; //用于快速周期性映射

  FILE *ti_result;

  int charge_pertubated = 0;

  float *h_charge;
  float *d_charge;
  float *h_charge_A;
  float *h_charge_B;
  float *d_charge_B_A;
  float *h_charge_B_A;
  int *h_subsys_division;
  int *d_subsys_division;

  struct cross_pme {
    float *PME_Q_B_A;
    float *d_cross_reciprocal_ene;
    float *d_cross_self_ene;
    float *charge_sum_B_A;
    float *d_cross_correction_atom_energy;
    float *d_cross_correction_ene;
    float *d_cross_direct_ene;
    float dH_dlambda;
    float cross_reciprocal_ene;
    float cross_self_ene;
    float cross_direct_ene;
    float cross_correction_ene;
    void Initial(const int atom_numbers, const int PME_Nall);
  } cross_pme;

  struct non_bond_information {
    float cutoff = 10.0;
    float skin = 2.0;
    int excluded_atom_numbers;  //排除表总长
    int *d_excluded_list_start; //记录每个原子的剔除表起点
    int *d_excluded_list;       //剔除表
    int *d_excluded_numbers;    //记录每个原子需要剔除的原子个数
    int *h_excluded_list_start; //记录每个原子的剔除表起点
    int *h_excluded_list;       //剔除表
    int *h_excluded_numbers;    //记录每个原子需要剔除的原子个数
    void Initial(CONTROLLER *controller, TI_CORE *TI_core);
  } nb; // 非键信息

  struct periodic_box_condition_information {
    VECTOR crd_to_uint_crd_cof;         //实坐标到整数坐标
    VECTOR quarter_crd_to_uint_crd_cof; //实坐标到0.25倍整数坐标
    VECTOR uint_dr_to_dr_cof;           //整数坐标到实坐标
    void Update_Volume(VECTOR box_length);
  } pbc;

  struct trajectory_input {
    TI_CORE *TI_core;
    int frame_numbers = 1000;
    int current_frame = 0;
    int bytes_per_frame = 0;
    FILE *crd_traj = NULL;
    // FILE * vel_traj = NULL;
    FILE *box_traj = NULL;
    void Initial(CONTROLLER *controller, TI_CORE *TI_core);
  } input; //轨迹读入信息

  struct dH_dlambda_data {
    float bondA_ene;
    float bondB_ene;
    float angleA_ene;
    float angleB_ene;
    float dihedralA_ene;
    float dihedralB_ene;
    float nb14A_EE_ene;
    float nb14A_LJ_ene;
    float nb14B_EE_ene;
    float nb14B_LJ_ene;

    float bond_soft_dH_dlambda;
    float lj_soft_dH_dlambda;
    float coul_direct_dH_dlambda;
    float lj_soft_long_range_correction;
    float pme_dH_dlambda;
    float pme_self_dH_dlambda;
    float pme_corr_dH_dlambda;
    float pme_reci_dH_dlambda;
    float kinetic_dH_dlambda;

    float dH_dlambda_current_frame;
    float total_dH_dlambda = 0;
    float average_dH_dlambda = 0;

    void Sum_One_Frame();
  } data;

  //用来将原子的真实坐标转换为unsigned
  //int坐标,注意factor需要乘以0.5（保证越界坐标自然映回box）
  void TI_Core_Crd_To_Uint_Crd();

  void Read_Next_Frame();

  void TI_Core_Crd_Device_To_Host();

  float Get_Cross_PME_Partial_H_Partial_Lambda(Particle_Mesh_Ewald *pme,
                                               const ATOM_GROUP *nl,
                                               int lj_pertubated,
                                               int is_download = 1);

  void Print_dH_dlambda_Average_To_Screen_And_Result_File();

  //释放空间
  void Clear();
};

#endif
