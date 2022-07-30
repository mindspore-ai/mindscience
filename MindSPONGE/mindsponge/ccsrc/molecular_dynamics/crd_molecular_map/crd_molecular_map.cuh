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


#ifndef COORDINATE_MOLECULAR_MAP
#define COORDINATE_MOLECULAR_MAP
#include "../common.cuh"
#include "../control.cuh"
#include <deque>

//20210420 分子映射目前是用排除表构建的——存在有排除表的原子会被视为属于同一个分子
struct CoordinateMolecularMap
{
    char module_name[CHAR_LENGTH_MAX];
    int is_initialized = 0;
    int last_modify_date = 20210830;

    //体系基本信息
    int atom_numbers=0;
    VECTOR box_length;

    //用于需要分子内坐标的解wrap的坐标记录
    VECTOR *nowrap_crd = NULL;
    VECTOR *old_crd = NULL;
    INT_VECTOR *box_map_times = NULL;
    VECTOR *h_nowrap_crd = NULL;
    VECTOR *h_old_crd = NULL;
    INT_VECTOR *h_box_map_times = NULL;

    int threads_per_block = 256;
    int blocks_per_grid = 20;
    //注意传入的crd是device上地址，一般初始化的时候总是有这个东西的
    void Initial(int atom_numbers, VECTOR box_length, VECTOR *crd, 
        const int exclude_numbers, const int *exclude_length, const int *exclude_start, const int *exclude_list, const char *module_name = NULL);
    //清除内存
    void Clear();

    //传入实时模拟中的crd坐标（GPU上），计算更新nowarp_crd，用于分子内坐标计算
    void Calculate_No_Wrap_Crd(const VECTOR *crd);
    //在每次有对原子进行周期性映射的操作后加入下面这个函数以更新跨盒子次数记录表
    void Refresh_BoxMapTimes(const VECTOR *crd);

    //CPU上函数，判断近邻表模块中进行box_map操作前后的old_crd,crd的穿越盒子信息box_map_times
    void Record_Box_Map_Times_Host(int atom_numbers, VECTOR *crd, VECTOR *old_crd, INT_VECTOR *box_map_times, VECTOR box);

    void Update_Volume(VECTOR box_length);

};
#endif //COORDINATE_MOLECULAR_MAP(crd_molecular_map.cuh)
