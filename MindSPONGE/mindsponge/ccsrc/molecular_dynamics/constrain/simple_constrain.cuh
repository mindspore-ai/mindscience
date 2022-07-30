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


#ifndef SIMPLE_CONSTARIN_CUH
#define SIMPLE_CONSTARIN_CUH
#include "../common.cuh"
#include "../control.cuh"
#include "constrain.cuh"

struct SIMPLE_CONSTRAIN
{
    char module_name[CHAR_LENGTH_MAX];
    int is_initialized = 0;
    int is_controller_printf_initialized = 0;
    int last_modify_date = 20211222;

    CONSTRAIN *constrain;

    //约束内力，使得主循环中更新后的坐标加上该力（力的方向与更新前的pair方向一致）修正，得到满足约束的坐标。
    VECTOR *constrain_frc = NULL;
    //每对的维里
    float *d_pair_virial = NULL;
    //总维里
    float *d_virial = NULL;
    //进行constrain迭代过程中的不断微调的原子uint坐标
    UNSIGNED_INT_VECTOR *test_uint_crd = NULL;
    //主循环中更新前的pair向量信息
    VECTOR *last_pair_dr = NULL;

    float step_length = 1.0f;//迭代求力时选取的步长，步长为1.可以刚好严格求得两体的constrain
                            //但对于三体及以上的情况，步长小一点会更稳定，但随之而来可能要求迭代次数增加
    int iteration_numbers = 25;//迭代步数
    
    //在加入各种constrain_pair后初始化
    //最后的exp_gamma为朗之万刘剑热浴的exp_gamma
    void Initial_Simple_Constrain
        (CONTROLLER *controller, CONSTRAIN *constrain, const char *module_name = NULL);
    
    //清除内存
    void Clear();
    //记录更新前的距离
    void Remember_Last_Coordinates(VECTOR *crd, UNSIGNED_INT_VECTOR* uint_crd, VECTOR scaler);
    //进行约束迭代
    void Constrain
        (VECTOR *crd, VECTOR *vel, const float *mass_inverse, const float *d_mass, VECTOR box_length, int need_pressure, float *d_pressure);
    //体积变化时的参数更新
    void Update_Volume(VECTOR box_length);

};



#endif //SIMPLE_CONSTARIN_CUH(simple_constrain.cuh)
