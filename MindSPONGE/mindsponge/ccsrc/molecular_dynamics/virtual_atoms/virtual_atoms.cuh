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


#ifndef VIRTUAL_ATOMS_CUH
#define VIRTUAL_ATOMS_CUH
#include "../control.cuh"
#include "../common.cuh"
#include <vector>

//virtual atom type 0
// x_v = x_1
// y_v = y_1
// z_v = 2 * h - z_1
//         1
//         ↑    
//         —    镜面   ↑
//         ↓     |     h
//         V     |     |
//            盒子底部 ↓
struct VIRTUAL_TYPE_0
{
    int virtual_atom;
    int from_1;
    float h_double;
};

struct VIRTUAL_TYPE_0_INFROMATION
{
    int virtual_numbers = 0;                                                                                                                                                                                
    VIRTUAL_TYPE_0 *h_virtual_type_0 = NULL;
    VIRTUAL_TYPE_0 *d_virtual_type_0 = NULL;
};

//virtual atom type 1
//r_v1 = a * r_21
//   1 - a - v - 1-a - 2
struct VIRTUAL_TYPE_1
{
    int virtual_atom;
    int from_1;
    int from_2;
    float a;
};

struct VIRTUAL_TYPE_1_INFROMATION
{
    int virtual_numbers = 0;                                                                                                                                                                                
    VIRTUAL_TYPE_1 *h_virtual_type_1 = NULL;
    VIRTUAL_TYPE_1 *d_virtual_type_1 = NULL;
};

//virtual atom type 2
//r_v1 = a * r_21 + b * r_31
struct VIRTUAL_TYPE_2
{
    int virtual_atom;
    int from_1;
    int from_2;
    int from_3;
    float a;
    float b;
};

struct VIRTUAL_TYPE_2_INFROMATION
{
    int virtual_numbers = 0;                                                                                                                                                                                
    VIRTUAL_TYPE_2 *h_virtual_type_2 = NULL;
    VIRTUAL_TYPE_2 *d_virtual_type_2 = NULL;
};

//virtual atom type 3
//r_v1 =  d * (r_12 + k * r_23)/|r_12 + k * r_23|
//           1
//           ↑  d
//           V  
//        ↙ |  ↘
//      2- k - 1-k -3
struct VIRTUAL_TYPE_3
{
    int virtual_atom;
    int from_1;
    int from_2;
    int from_3;
    float d;
    float k;
};

struct VIRTUAL_TYPE_3_INFROMATION
{
    int virtual_numbers = 0;                                                                                                                                                                                
    VIRTUAL_TYPE_3 *h_virtual_type_3 = NULL;
    VIRTUAL_TYPE_3 *d_virtual_type_3 = NULL;
};


struct VIRTUAL_LAYER_INFORMATION
{
    VIRTUAL_TYPE_0_INFROMATION v0_info;
    VIRTUAL_TYPE_1_INFROMATION v1_info;
    VIRTUAL_TYPE_2_INFROMATION v2_info;
    VIRTUAL_TYPE_3_INFROMATION v3_info;
};

struct VIRTUAL_INFORMATION
{
    //模块信息
    char module_name[CHAR_LENGTH_MAX];
    int is_initialized = 0;
    int is_controller_printf_initialized = 0;
    int last_modify_date = 20210830;

    //cuda信息
    int threads_per_block = 128;

    //内容信息
    int max_level = 0; //最大的虚拟层级
    
    int *virtual_level = NULL; //每个原子的虚拟位点层级：0->实原子，1->只依赖于实原子，2->依赖的原子的虚拟等级最高为1，以此类推...
    
    std::vector<VIRTUAL_LAYER_INFORMATION> virtual_layer_info; //记录每个层级的信息
    
    void Initial(CONTROLLER *controller, int atom_numbers, int *system_freedom, const char *module_name = NULL);  //初始化
    void Clear();  //清除内存
    void Force_Redistribute(const UNSIGNED_INT_VECTOR *uint_crd, const VECTOR scaler, VECTOR *force);  //进行力重分配
    
    void Coordinate_Refresh(const UNSIGNED_INT_VECTOR *uint_crd, const VECTOR scaler, VECTOR *crd);    //更新虚拟位点的坐标
};
#endif
