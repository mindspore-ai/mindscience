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


#ifndef FEP_CORE_CUH
#define FEP_CORE_CUH
#include "../common.cuh"
#include "../control.cuh"

struct partition_energy_data
{
    float bond_ene;
    float angle_ene;
    float dihedral_ene;
    float nb14_LJ_ene;
    float nb14_EE_ene;

    float bond_soft_ene;
    float vdw_intersys_ene;
    float vdw_intrasys_ene;
    float coul_direct_intersys_ene;
    float coul_direct_intrasys_ene;
    float vdw_long_range_correction;
    float coul_long_range;

    float pV;
};

//普通分子模拟所涉及的大部分信息
struct FEP_CORE
{
    int is_Initialized = 0;
    int last_modify_date = 20210525;

    //sponge输入初始化
    void Initial(CONTROLLER *controller);

    int atom_numbers = 0;//模拟的总原子数目
    VECTOR box_length;
    VECTOR last_box_length;
    float volume_change_factor;
    VECTOR box_angle;
    //每个原子的基本物理测量量，on host
    VECTOR *coordinate = NULL;

    //每个原子的基本物理测量量，on device
    VECTOR *crd = NULL;
    UNSIGNED_INT_VECTOR *uint_crd = NULL;//用于快速周期性映射

    int charge_pertubated = 0;

    float * h_charge;
    float * d_charge;
    int * h_subsys_division;
    int * d_subsys_division;

    float *d_direct_atom_energy_intersys = NULL;
    float *d_direct_atom_energy_intrasys = NULL;
    float *d_direct_ene_intersys = NULL;
    float *d_direct_ene_intrasys = NULL;

    FILE * float32ene_file;

    struct non_bond_information
    {
        float cutoff = 10.0;
        float skin = 2.0;
        int excluded_atom_numbers; //排除表总长
        int *d_excluded_list_start;//记录每个原子的剔除表起点
        int *d_excluded_list;//剔除表
        int *d_excluded_numbers;//记录每个原子需要剔除的原子个数
        int *h_excluded_list_start;//记录每个原子的剔除表起点
        int *h_excluded_list;//剔除表
        int *h_excluded_numbers;//记录每个原子需要剔除的原子个数
        void Initial(CONTROLLER *controller, FEP_CORE *FEP_core);    
    } nb; // 非键信息

    struct periodic_box_condition_information
    {
        VECTOR crd_to_uint_crd_cof;//实坐标到整数坐标
        VECTOR quarter_crd_to_uint_crd_cof;//实坐标到0.25倍整数坐标
        VECTOR uint_dr_to_dr_cof;//整数坐标到实坐标
        void Update_Volume(VECTOR box_length);
    } pbc;

    struct trajectory_input
    {
        FEP_CORE * FEP_core;
        int frame_numbers = 1000;
        int current_frame = 0;
        int bytes_per_frame = 0;
        FILE * crd_traj = NULL;
        FILE * box_traj = NULL;
        void Initial(CONTROLLER * controller, FEP_CORE * FEP_core);
    } input; //轨迹读入信息

    struct energy_data
    {
        partition_energy_data partition;

        float current_frame_ene;
        float pressure;
        float temperature;
        
        float lj_soft_ene;
        float * frame_ene;
        partition_energy_data * frame_partition_ene;

        void Sum_One_Frame(int current_frame);
    }data;

    //用来将原子的真实坐标转换为unsigned int坐标,注意factor需要乘以0.5（保证越界坐标自然映回box）
    void FEP_Core_Crd_To_Uint_Crd();
    
    void Read_Next_Frame();

    void FEP_Core_Crd_Device_To_Host();

    void Print_Pure_Ene_To_Result_File();

    void Seperate_Direct_Atom_Energy(ATOM_GROUP * nl, const float pme_beta);

    //释放空间
    void Clear();
};

#endif
