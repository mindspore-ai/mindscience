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


#ifndef NOSE_HOOVER_CHAIN_CUH
#define NOSE_HOOVER_CHAIN_CUH
#include "../common.cuh"
#include "../control.cuh"


//用于记录与计算Nose-Hoover链控温相关的信息
struct NOSE_HOOVER_CHAIN_INFORMATION
{
    char module_name[CHAR_LENGTH_MAX];
    int is_initialized = 0;
    int is_controller_printf_initialized = 0;
    int last_modify_date = 20211101;

    
    int chain_length = 0;//NH链长度
    float *coordinate = NULL; //拓展自由度的坐标
    float *velocity = NULL; //拓展自由度的速度
    float h_mass; //拓展自由度的质量
    float kB_T = 0;

    float max_velocity = 0;   //最大速度
    char restart_file_name[CHAR_LENGTH_MAX]; //重启文件名字
    FILE *f_crd_traj = NULL, *f_vel_traj = NULL; //坐标和速度轨迹文件


    //初始化
    void Initial(CONTROLLER *controller, float target_pressure, const char *module_name = NULL);
    

    void MD_Iteration_Leap_Frog(int atom_numbers, VECTOR *vel, VECTOR *crd, VECTOR *frc, VECTOR *acc, float *inverse_mass, float dt, float Ek, int freedom);

    void Save_Restart_File();

    void Save_Trajectory_File();
};

#endif //ANDERSEN_THERMOSTAT_CUH(Anderson.cuh)
