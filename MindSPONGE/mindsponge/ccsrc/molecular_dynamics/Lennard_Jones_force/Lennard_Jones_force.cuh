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


#ifndef LENNARD_JONES_FORCE_CUH
#define LENNARD_JONES_FORCE_CUH
#include "../common.cuh"
#include "../control.cuh"

//用于计算LJ_Force时使用的坐标和记录的原子LJ种类序号与原子电荷
#ifndef UINT_VECTOR_LJ_TYPE_DEFINE
#define UINT_VECTOR_LJ_TYPE_DEFINE
struct UINT_VECTOR_LJ_TYPE
{
    unsigned int uint_x;
    unsigned int uint_y;
    unsigned int uint_z;
    int LJ_type;
    float charge;
};
__device__ __host__ VECTOR Get_Periodic_Displacement(const UINT_VECTOR_LJ_TYPE uvec_a, const UINT_VECTOR_LJ_TYPE uvec_b, const VECTOR scaler);
__global__ void Copy_LJ_Type_To_New_Crd(const int atom_numbers, UINT_VECTOR_LJ_TYPE *new_crd, const int *LJ_type);
__global__ void Copy_Crd_And_Charge_To_New_Crd(const int atom_numbers, const UNSIGNED_INT_VECTOR *crd, UINT_VECTOR_LJ_TYPE *new_crd, const float *charge);
__global__ void Copy_Crd_To_New_Crd(const int atom_numbers, const UNSIGNED_INT_VECTOR *crd, UINT_VECTOR_LJ_TYPE *new_crd);
#endif

//用于记录与计算LJ相关的信息
struct LENNARD_JONES_INFORMATION
{
    char module_name[CHAR_LENGTH_MAX];
    int is_initialized = 0;
    int is_controller_printf_initialized = 0;
    int last_modify_date = 20210830;

    //a = LJ_A between atom[i] and atom[j]
    //b = LJ_B between atom[i] and atom[j]
    //E_lj = a/12 * r^-12 - b/6 * r^-6;
    //F_lj = (a * r^-14 - b * r ^ -6) * dr
    int atom_numbers = 0;           //原子数
    int atom_type_numbers = 0;      //原子种类数
    int pair_type_numbers = 0;      //原子对种类数
    

    int *h_atom_LJ_type = NULL;        //原子对应的LJ种类
    int *d_atom_LJ_type = NULL;        //原子对应的LJ种类
    
    float *h_LJ_A = NULL;              //LJ的A系数
    float *h_LJ_B = NULL;              //LJ的B系数
    float *d_LJ_A = NULL;              //LJ的A系数
    float *d_LJ_B = NULL;              //LJ的B系数
    
    float *h_LJ_energy_atom = NULL;    //每个原子的LJ的能量
    float h_LJ_energy_sum = 0;     //所有原子的LJ能量和
    float *d_LJ_energy_atom = NULL;    //每个原子的LJ的能量
    float *d_LJ_energy_sum = NULL;     //所有原子的LJ能量和

    dim3 thread_LJ = { 32, 32 }; // cuda参数
    //初始化
    void Initial(CONTROLLER *controller, float cutoff, VECTOR box_length, const char *module_name = NULL);
    //从amber的parm文件里读取
    void Initial_From_AMBER_Parm(const char *file_name, CONTROLLER controller);
    //清除内存
    void Clear();
    //分配内存
    void LJ_Malloc();
    //参数传到GPU上
    void Parameter_Host_To_Device();
    

    float cutoff = 10.0;
    VECTOR uint_dr_to_dr_cof;
    float volume = 0;
    UINT_VECTOR_LJ_TYPE *uint_crd_with_LJ = NULL;

    
    //可以根据外界传入的need_atom_energy和need_virial，选择性计算能量和维里。其中的维里对PME直接部分计算的原子能量，在和PME其他部分加和后即维里。
    void LJ_PME_Direct_Force_With_Atom_Energy_And_Virial(const int atom_numbers, const UNSIGNED_INT_VECTOR *uint_crd, const float *charge, VECTOR *frc,
        const ATOM_GROUP *nl, const float pme_beta, const int need_atom_energy, float *atom_energy,
        const int need_virial, float *atom_lj_virial, float *atom_direct_pme_energy);

    //求力的时候对能量和维里的长程修正
    void Long_Range_Correction(int need_pressure, float *d_virial, int need_potential, float *d_potential);

    //获得能量
    float Get_Energy(const UNSIGNED_INT_VECTOR *uint_crd, const ATOM_GROUP *nl, int is_download = 1);

    //更新体积
    void Update_Volume(VECTOR box_length);
    /*-----------------------------------------------------------------------------------------
    下面的函数是其他需求的排列组合，但是接口没有特地优化，如果自己需要，可能需要修改接口或写一个重载函数
    ------------------------------------------------------------------------------------------*/

    
    //最原始优化版本的纯粹Lennard_Jones力的计算，采用多thread对单个原子同时计算。
    //（总原子数，近邻表，近邻表中原子个数，加上原子LJ种类的原子坐标结构体，整数坐标到实坐标映射参数，
    //LJ种类的A系数（已乘12），B系数（已乘6），截断半径平方，原子受到的力）
    void LJ_Force(const UINT_VECTOR_LJ_TYPE *uint_crd, const VECTOR scaler, VECTOR *frc,
        const ATOM_GROUP *nl, const float cutoff_square);
    //用于计算LJ的作用能量参数基本仿照LJ_Force函数
    void LJ_Energy(const UNSIGNED_INT_VECTOR *uint_crd, const ATOM_GROUP *nl);
    //给SITS使用计算总LJ能的函数
    void LJ_Energy(const UNSIGNED_INT_VECTOR *uint_crd, const ATOM_GROUP *nl, const int is_reset, const int is_sum);

    //在上述函数基础上，加上了与PME方法相匹配的Direct静电力部分，需要额外提供pme_beta参数，和常数2/sqrt(pi)
    //且需要注意cutoff不用再平方，UINT_VECTOR_LJ_TYPE中已经含有charge参数
    //二维thread循环方式经过调整，效率更高了2020年7月21日
    //这里的atom_numbers参数可以小于等于体系真正的原子数，这样可以只计算部分的LJ作用，在SITS和表面加速的方法中会用到
    void LJ_Force_With_PME_Direct_Force(const int atom_numbers, const UINT_VECTOR_LJ_TYPE *uint_crd, const VECTOR scaler, VECTOR *frc,
        const ATOM_GROUP *nl, const float cutoff, const float pme_beta);//实际是LJ_Force_With_PME_Direct_Force的一个重载，将atom_numbers暴露出来

    void LJ_Force_With_PME_Direct_Force(const UNSIGNED_INT_VECTOR *uint_crd, const float *charge, VECTOR *frc, const ATOM_GROUP *nl, const float pme_beta);


    //这里的atom_numbers参数可以小于等于体系真正的原子数，这样可以只计算部分的LJ作用，在SITS和表面加速的方法中会用到
    //这个记录能量列表也从外部传入
    void LJ_Energy(const int atom_numbers, const UINT_VECTOR_LJ_TYPE *uint_crd, const VECTOR scaler,
        const ATOM_GROUP *nl, const float cutoff_square, float *d_LJ_energy_atom);//直接是LJ_Energy的一个重载，将d_LJ_energy_atom暴露出来
    void LJ_PME_Direct_Force_With_Atom_Energy(const int atom_numbers, const UINT_VECTOR_LJ_TYPE *uint_crd, const VECTOR scaler, VECTOR *frc,
        const ATOM_GROUP *nl, const float cutoff, const float pme_beta,float *atom_energy);


    
    //上面的函数去掉PME直接部分的版本
    void LJ_Force_With_Atom_Energy_And_Virial(const int atom_numbers, const UINT_VECTOR_LJ_TYPE *uint_crd, const VECTOR scaler, VECTOR *frc,
        const ATOM_GROUP *nl, const float cutoff, int need_atom_energy, float *atom_energy,
        int need_virial, float *atom_lj_virial, float *virial);


    //长程能量和维里修正
    float long_range_factor = 0;
    //求分类的能量的时候算的
    void Long_Range_Correction(float volume);

    //能量从GPU到CPU
    void Energy_Device_To_Host();
};
#endif //LENNARD_JONES_FORCE_CUH(Lennard_Jones_force.cuh)
