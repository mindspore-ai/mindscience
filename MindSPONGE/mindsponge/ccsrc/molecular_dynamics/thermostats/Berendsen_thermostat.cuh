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


#ifndef BERENDSEN_THERMOSTAT_CUH
#define BERENDSEN_THERMOSTAT_CUH
#include "../common.cuh"
#include "../control.cuh"
#include <random>


//用于记录与计算Berendsen控温相关的信息
struct BERENDSEN_THERMOSTAT_INFORMATION
{
    char module_name[CHAR_LENGTH_MAX];
    int is_initialized = 0;
    int is_controller_printf_initialized = 0;
    int last_modify_date = 20211101;

    //原始版本berendsen控温
    float tauT; //弛豫时间（ps）
    float dt;  //步长（ps）
    float target_temperature; //目标温度
    float lambda; //规度系数

    //Bussi的修正
    //文献：Canonical sampling through velocity-rescaling
    int stochastic_term = 0;
    std::default_random_engine e;
    std::normal_distribution<float> n;

    //初始化
    void Initial(CONTROLLER *controller, float target_temperature, const char *module_name = NULL);
    
    void Scale_Velocity(int atom_numbers, VECTOR *vel);

    void Record_Temperature(float temperature, int freedom);
};

#endif //BERENDSEN_THERMOSTAT_CUH(Berndsen.cuh)
