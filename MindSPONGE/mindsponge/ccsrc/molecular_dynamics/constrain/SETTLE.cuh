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


#ifndef SETTLE_CUH
#define SETTLE_CUH
#include "../common.cuh"
#include "../control.cuh"
#include "constrain.cuh"

struct CONSTRAIN_TRIANGLE
{
    int atom_A;
    int atom_B;
    int atom_C;
    float ra;
    float rb;
    float rc;
    float rd;
    float re;
};


struct SETTLE
{
    char module_name[CHAR_LENGTH_MAX];
    int is_initialized = 0;
    int is_controller_printf_initialized = 0;
    int last_modify_date = 20211222;

    CONSTRAIN *constrain;

    void Initial(CONTROLLER* controller, CONSTRAIN* constrain, float *h_mass, const char* module_name = NULL);

    int triangle_numbers = 0;
    CONSTRAIN_TRIANGLE* d_triangles = NULL, *h_triangles = NULL;

    int pair_numbers = 0;
    CONSTRAIN_PAIR *d_pairs = NULL, *h_pairs = NULL;

    VECTOR* last_pair_AB = NULL;
    VECTOR* last_triangle_BA = NULL;
    VECTOR* last_triangle_CA = NULL;
    void Remember_Last_Coordinates(UNSIGNED_INT_VECTOR* uint_crd, VECTOR scaler);

    float* virial = NULL;
    VECTOR* virial_vector = NULL;
    void Do_SETTLE(const float* d_mass, VECTOR* crd, VECTOR box_length, VECTOR* vel,
        int need_pressure, float* d_pressure);
};



#endif //SETTLE_CUH(settle.cuh)
