#ifndef MAIN_RUN_CUH
#define MAIN_RUN_CUH


#include "common.cuh"
#include "control.cuh"
#include "MD_core/MD_core.cuh"
#include "bond/bond.cuh"
#include "angle/angle.cuh"
#include "angle/Urey_Bradley_force.cuh"
#include "dihedral/dihedral.cuh"
#include "dihedral/improper_dihedral.cuh"
#include "nb14/nb14.cuh"
#include "cmap/cmap.cuh"
#include "neighbor_list/neighbor_list.cuh"
#include "Lennard_Jones_force/Lennard_Jones_force.cuh"
#include "PME_force/PME_force.cuh"
#include "thermostats/Middle_Langevin_MD.cuh"
#include "thermostats/Langevin_MD.cuh"
#include "thermostats/Andersen_thermostat.cuh"
#include "thermostats/Berendsen_thermostat.cuh"
#include "thermostats/nose_hoover_chain.cuh"
#include "barostats/MC_barostat.cuh"
#include "barostats/Berendsen_barostat.cuh"
#include "barostats/andersen_barostat.cuh"
#include "restrain/restrain.cuh"
#include "constrain/constrain.cuh"
#include "constrain/SETTLE.cuh"
#include "constrain/SHAKE.cuh"
#include "constrain/simple_constrain.cuh"
#include "virtual_atoms/virtual_atoms.cuh"
#include "crd_molecular_map/crd_molecular_map.cuh"
#include "Lennard_Jones_force/LJ_soft_core.cuh"
#include "bond/bond_soft.cuh"


void Main_Initial(int argc, char *argv[]);
void Main_Clear();

void Main_Calculate_Force();
void Main_Iteration();
void Main_Print();

void Main_Volume_Change(double factor);
void Main_Box_Length_Change(VECTOR factor);
void Main_Volume_Change_Largely();

#endif //MAIN_CUH
