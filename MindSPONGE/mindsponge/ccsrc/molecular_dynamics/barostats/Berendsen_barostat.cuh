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


#ifndef BERENDSEN_BAROSTAT_CUH
#define BERENDSEN_BAROSTAT_CUH
#include "../common.cuh"
#include "../control.cuh"
#include <random>

//用于记录与计算Berendsen控压相关的信息
struct BERENDSEN_BAROSTAT_INFORMATION
{
    char module_name[CHAR_LENGTH_MAX];
    int is_initialized = 0;
    int is_controller_printf_initialized = 0;
    int last_modify_date = 20210825;


    float taup; //压强弛豫时间（ps）
    float dt;  //步长（ps）
    int update_interval; //更新间隔
    float compressibility; //压缩系数（mdin里单位是bar^-1, 转换存为程序内单位）

    float V0; //体积
    float newV; //新体积

    double crd_scale_factor;     //坐标系数因子

    //随机修正
    //文献：Pressure control using stochastic cell rescaling
    int stochastic_term = 0;
    std::default_random_engine e;
    std::normal_distribution<float> n;

    //初始化
    void Initial(CONTROLLER *controller, float target_pressure, VECTOR box_length, const char *module_name = NULL);


    void Ask_For_Calculate_Pressure(int steps, int *need_pressure);


};

#endif //BERENDSEN_BAROSTAT_CUH(Berndsen.cuh)
