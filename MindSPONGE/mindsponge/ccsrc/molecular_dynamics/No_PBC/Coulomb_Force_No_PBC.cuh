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


#ifndef COULOMB_FORCE_NO_PBC_CUH
#define COULOMB_FORCE_NO_PBC_CUH
#include "../common.cuh"
#include "../control.cuh"

//用于记录与计算LJ相关的信息
struct COULOMB_FORCE_NO_PBC_INFORMATION
{
    char module_name[CHAR_LENGTH_MAX];
    int is_initialized = 0;
    int is_controller_printf_initialized = 0;
    int last_modify_date = 20220603;

    //E_lj = qi * qj / r;
    //F_lj = qi * qj / r ^ 2;
    int atom_numbers = 0;           //原子数

    float *h_Coulomb_energy_atom = NULL;    //每个原子的Coulomb的能量
    float h_Coulomb_energy_sum = 0;     //所有原子的Coulomb能量和
    float *d_Coulomb_energy_atom = NULL;    //每个原子的Coulomb的能量
    float *d_Coulomb_energy_sum = NULL;     //所有原子的Coulomb能量和

    dim3 thread_Coulomb = { 32, 32 };

    //初始化
    void Initial(CONTROLLER *controller, int atom_numbers, float cutoff, const char *module_name = NULL);
    //清除内存
    void Clear();
    //分配内存
    void Malloc();

    float cutoff = 10.0;

    //可以根据外界传入的need_atom_energ选择性计算能量
    void Coulomb_Force_With_Atom_Energy(const int atom_numbers, const VECTOR *crd, const float *charge, VECTOR *frc, const int need_atom_energy, float *atom_energy,
        const int *excluded_list_start, const int *excluded_list, const int *excluded_atom_numbers);

    //获得能量
    float Get_Energy(const VECTOR *crd, const float *charge, const int *excluded_list_start, const int *excluded_list, const int *excluded_atom_numbers, int is_download = 1);
};
#endif //LENNARD_JONES_FORCE_CUH(Lennard_Jones_force.cuh)
