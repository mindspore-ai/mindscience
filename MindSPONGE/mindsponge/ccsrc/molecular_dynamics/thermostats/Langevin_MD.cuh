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


#ifndef Langevin_MD_CUH
#define Langevin_MD_CUH
#include "../common.cuh"
#include "../control.cuh"

struct Langevin_MD_INFORMATION
{
    char module_name[CHAR_LENGTH_MAX];
    int is_initialized = 0;
    int is_controller_printf_initialized = 0;
    int last_modify_date = 20210825;

    int threads_per_block = 128;


    float max_velocity = 0;
    int atom_numbers;
    float dt;
    float half_dt;
    float gamma_ln = 1.0;//碰撞频率
    float target_temperature = 300.0;//热浴温度

    float sigma_ln;//朗之万动力学中参数(=sqrtf(2.*gamma_ln*KB*temp0 / dt))
    int float4_numbers;//存储随机数的长度
    curandStatePhilox4_32_10_t *rand_state = NULL;//用于记录随机数发生器状态
    VECTOR *random_force = NULL;//存储随机力矢量，要求该数组的长度要能整除4且大于等于atom_numbers
    float *h_sigma_mass = NULL;//用于朗之万热浴过程中随机力的原子等效质量
    float *d_sigma_mass = NULL;//用于朗之万热浴过程中随机力的原子等效质量
    float *d_mass_inverse = NULL; //质量的倒数
    //初始化（质量信息从某个MD_CORE的已初始化质量数组中获得）
    void Initial(CONTROLLER *controller, const int atom_numbers, const float target_temperature, const float *h_mass, const char *module_name = NULL);
//清除内存
    void Clear();
    //迭代算法
    void MD_Iteration_Leap_Frog(VECTOR *frc, VECTOR *crd, VECTOR *vel, VECTOR *acc);
};

#endif //Langevin_MD_CUH(Langevin_MD.cuh)
