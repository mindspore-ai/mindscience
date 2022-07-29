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


#ifndef RESTRAIN_CUH
#define RESTRAIN_CUH
#include "../common.cuh"
#include "../control.cuh"

struct RESTRAIN_INFORMATION 
{
    char module_name[CHAR_LENGTH_MAX];
    int is_initialized = 0;
    int is_controller_printf_initialized = 0;
    int last_modify_date = 20210831;

    //E_restrain = 0.5 * weight * (r - r_ref) ^ 2
    int restrain_numbers = 0; //限制的原子数量
    int *h_lists = NULL; //限制的原子序号
    int *d_lists = NULL; //限制的原子序号
    
    float *d_restrain_ene = NULL;
    float h_sum_of_restrain_ene;
    float *d_sum_of_restrain_ene = NULL;

    float weight = 20; //限制力常数
    VECTOR *crd_ref = NULL; //限制的参考坐标(在GPU上)

    //cuda计算分配相关参数
    int threads_per_block = 128;

    //Restrain初始化(总原子数，GPU上所有原子的坐标，控制，模块名)
    void Initial(CONTROLLER *control, const int atom_numbers, const VECTOR *crd, const char *module_name = NULL);
    //清空模块
    void Clear();

    //计算Restrain的能量、力和维里
    void Restraint(const VECTOR *crd, const VECTOR box_length,
        float *atom_energy, float *atom_virial, VECTOR *frc);

    //获得能量
    float Get_Energy(const VECTOR *crd, const VECTOR box_length, int is_download = 1);

};

#endif //RESTRAIN_CUH(restrain.cuh)
