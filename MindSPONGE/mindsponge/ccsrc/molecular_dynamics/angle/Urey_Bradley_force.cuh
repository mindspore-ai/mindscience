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

//20210901
/*
* 由于时间关系，该模块目前没经过严格且全面的验证
* 只能初始化一次，未加入clear等函数，不要反复初始化
* 
*/

#ifndef UREY_BRADLEY_FORCE_CUH
#define UREY_BRADLEY_FORCE_CUH
#include "../bond/bond.cuh"
#include "../angle/angle.cuh"
struct UREY_BRADLEY
{
    char module_name[CHAR_LENGTH_MAX];
    int is_initialized = 0;
    int is_controller_printf_initialized = 0;
    int last_modify_date = 20210901;

    BOND bond;
    ANGLE angle;

    int Urey_Bradley_numbers=0;

    void Initial(CONTROLLER* controller, char* module_name = NULL);
    void Urey_Bradley_Force_With_Atom_Energy_And_Virial
    (const UNSIGNED_INT_VECTOR* uint_crd, const VECTOR scaler, VECTOR* frc, float* atom_energy, float* atom_virial);
    float Get_Energy(const UNSIGNED_INT_VECTOR* uint_crd, const VECTOR scaler, int is_download = 1);
};

#endif //UREY_BRADLEY_FORCE_CUH(Urey_Bradley_force.cuh)