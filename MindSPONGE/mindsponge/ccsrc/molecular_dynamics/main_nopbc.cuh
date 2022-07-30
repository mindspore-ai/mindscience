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

#include "thermostats/Middle_Langevin_MD.cuh"
#include "thermostats/Langevin_MD.cuh"
#include "thermostats/Andersen_thermostat.cuh"
#include "thermostats/Berendsen_thermostat.cuh"
#include "thermostats/nose_hoover_chain.cuh"

#include "constrain/constrain.cuh"
#include "constrain/SETTLE.cuh"
#include "constrain/SHAKE.cuh"
#include "constrain/simple_constrain.cuh"
#include "virtual_atoms/virtual_atoms.cuh"
#include "crd_molecular_map/crd_molecular_map.cuh"

#include "restrain/restrain.cuh"

#include "No_PBC/Lennard_Jones_force_No_PBC.cuh"
#include "No_PBC/Coulomb_Force_No_PBC.cuh"
#include "No_PBC/generalized_Born.cuh"

void Main_Initial(int argc, char *argv[]);
void Main_Clear();

void Main_Calculate_Force();
void Main_Iteration();
void Main_Print();

#endif //MAIN_CUH
