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

#ifndef MAIN_FEP_CUH
#define MAIN_FEP_CUH

#include "FEP_core/FEP_core.cuh"
#include "Lennard_Jones_force/LJ_soft_core.cuh"
#include "PME_force/PME_force.cuh"
#include "angle/angle.cuh"
#include "bond/bond.cuh"
#include "bond/bond_soft.cuh"
#include "common.cuh"
#include "control.cuh"
#include "dihedral/dihedral.cuh"
#include "nb14/nb14.cuh"
#include "neighbor_list/neighbor_list.cuh"

void Main_Initial(int argc, char *argv[]);
void Main_Iteration();
void Main_Print();
void Main_Calculation();
void Main_Volume_Update();
void Main_Clear();

#endif
