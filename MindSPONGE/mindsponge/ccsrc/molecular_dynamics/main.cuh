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

#ifndef MAIN_RUN_CUH
#define MAIN_RUN_CUH

#include "Lennard_Jones_force/LJ_soft_core.cuh"
#include "Lennard_Jones_force/Lennard_Jones_force.cuh"
#include "MD_core/MD_core.cuh"
#include "PME_force/PME_force.cuh"
#include "angle/Urey_Bradley_force.cuh"
#include "angle/angle.cuh"
#include "barostats/Berendsen_barostat.cuh"
#include "barostats/MC_barostat.cuh"
#include "barostats/andersen_barostat.cuh"
#include "bond/bond.cuh"
#include "bond/bond_soft.cuh"
#include "cmap/cmap.cuh"
#include "common.cuh"
#include "constrain/SETTLE.cuh"
#include "constrain/SHAKE.cuh"
#include "constrain/constrain.cuh"
#include "constrain/simple_constrain.cuh"
#include "control.cuh"
#include "crd_molecular_map/crd_molecular_map.cuh"
#include "dihedral/dihedral.cuh"
#include "dihedral/improper_dihedral.cuh"
#include "nb14/nb14.cuh"
#include "neighbor_list/neighbor_list.cuh"
#include "restrain/restrain.cuh"
#include "thermostats/Andersen_thermostat.cuh"
#include "thermostats/Berendsen_thermostat.cuh"
#include "thermostats/Langevin_MD.cuh"
#include "thermostats/Middle_Langevin_MD.cuh"
#include "thermostats/nose_hoover_chain.cuh"
#include "virtual_atoms/virtual_atoms.cuh"

void Main_Initial(int argc, char *argv[]);
void Main_Clear();

void Main_Calculate_Force();
void Main_Iteration();
void Main_Print();

void Main_Volume_Change(double factor);
void Main_Box_Length_Change(VECTOR factor);
void Main_Volume_Change_Largely();

#endif // MAIN_CUH
