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


#ifndef CONSTARIN_CUH
#define CONSTARIN_CUH
#include "../common.cuh"
#include "../control.cuh"

struct CONSTRAIN_PAIR
{
    int atom_i_serial;
    int atom_j_serial;
    float constant_r;
    float constrain_k;//这个并不是说有个弹性系数来固定，而是迭代时，有个系数k=m1*m2/(m1+m2)
};

struct CONSTRAIN
{
    char module_name[CHAR_LENGTH_MAX];
    int is_initialized = 0;
    int is_controller_printf_initialized = 0;
    int last_modify_date = 20211222;

    int atom_numbers = 0;
    float dt = 0.001f;
    float dt_inverse;
    VECTOR uint_dr_to_dr_cof;//该系数可将无符号整型坐标之差变为实际坐标之差
    VECTOR quarter_crd_to_uint_crd_cof;//该系数可将实际坐标变为对应的一半长度的无符号整型坐标
    float volume; //体积

    float v_factor = 1.0f;  //一个积分步中,一个微小的力F对速度的影响，即dv = v_factor * F * dt/m
    float x_factor = 1.0f;  //一个积分步中,一个微小的力F对位移的影响，即dx = x_factor * F * dt * dt/m 
    float constrain_mass = 3.3;//对质量小于该值的原子进行限制

    //在初始化的时候用到，在实际计算中不会使用,在初始化时已经被释放
    int bond_constrain_pair_numbers = 0;
    int angle_constrain_pair_numbers = 0;
    CONSTRAIN_PAIR *h_bond_pair = NULL;
    CONSTRAIN_PAIR *h_angle_pair = NULL;

    //在实际计算中使用，体系总的constrain pair
    int constrain_pair_numbers = 0;
    CONSTRAIN_PAIR *constrain_pair = NULL;
    CONSTRAIN_PAIR *h_constrain_pair = NULL;

    //用于暂时记录bond的信息，便于angle中搜索bond长度
    //这些指针指向的空间并不由本模块申请且不由本模块释放
    struct BOND_INFORMATION
    {
        int bond_numbers;
        const int *atom_a = NULL;
        const int *atom_b = NULL;
        const float *bond_r = NULL;
    }bond_info;


    //默认的Initial需要按照下面的顺序：
    //Add_HBond_To_Constrain_Pair
    //Add_HAngle_To_Constrain_Pair
    //Initial_Constrain

    //20201125 由于MD_INFORMATION里面暂时没有加入原子序数，所以不用原子序数判断H，而是直接比较质量
    //当质量较小时，就认为是H
    //传入的指针指向HOST内存
    void Add_HBond_To_Constrain_Pair
        (CONTROLLER *controller, const int bond_numbers, const int *atom_a, const int *atom_b, const float *bond_r,
        const float *atom_mass, const char *module_name = NULL);//要求均是指向host上内存的指针

    //需要在先运行Add_HBond_To_Constrain_Pair之后再运行
    //传入的指针指向HOST内存
    void Add_HAngle_To_Constrain_Pair
        (CONTROLLER *controller, const int angle_numbers, const int *atom_a, const int *atom_b, const int *atom_c,
        const float *angle_theta, const float *atom_mass);//要求均是指向host上内存的指针
    
    //在加入各种constrain_pair后初始化
    //中间的exp_gamma为朗之万刘剑热浴的exp_gamma
    void Initial_Constrain
        (CONTROLLER *controller, const int atom_numbers, const float dt, const VECTOR box_length, const float exp_gamma, const int is_Minimization, float *atom_mass, int *system_freedom);
    
    //清除内存
    void Clear();
    //记录更新前的距离

    void Update_Volume(VECTOR box_length);

};



#endif //SIMPLE_CONSTARIN_CUH(simple_constrain.cuh)
