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


#ifndef ANDERSEN_THERMOSTAT_CUH
#define ANDERSEN_THERMOSTAT_CUH
#include "../common.cuh"
#include "../control.cuh"


//用于记录与计算Andersen控温相关的信息
struct ANDERSEN_THERMOSTAT_INFORMATION
{
    char module_name[CHAR_LENGTH_MAX];
    int is_initialized = 0;
    int is_controller_printf_initialized = 0;
    int last_modify_date = 20211101;

    //更新间隔
    int update_interval = 0;
    float max_velocity = 0;

    //高斯随机数相关
    int float4_numbers;//存储随机数的长度
    curandStatePhilox4_32_10_t *rand_state = NULL;//用于记录随机数发生器状态
    VECTOR *random_vel = NULL;//存储随机速度矢量，要求该数组的长度要能整除4且大于等于atom_numbers



    //温度相关的系数
    float *h_factor, *d_factor;

    //初始化
    void Initial(CONTROLLER *controller, float target_pressure, int atom_numbers, float *h_mass, const char *module_name = NULL);
    

    void MD_Iteration_Leap_Frog(int atom_numbers, VECTOR *vel, VECTOR *crd, VECTOR *frc, VECTOR *acc, float *inverse_mass, float dt);

    
};

#endif //ANDERSEN_THERMOSTAT_CUH(Anderson.cuh)
