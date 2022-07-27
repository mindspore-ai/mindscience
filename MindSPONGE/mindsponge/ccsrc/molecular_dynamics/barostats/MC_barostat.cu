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

#include "MC_barostat.cuh"

static __global__ void Scale_Vector_Atomically(int n, VECTOR *vlist,
                                               VECTOR scaler) {
  VECTOR vtemp;
  for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    vtemp = vlist[i];
    vtemp.x *= scaler.x;
    vtemp.y *= scaler.y;
    vtemp.z *= scaler.z;
    vlist[i] = vtemp;
  }
}

void MC_BAROSTAT_INFORMATION::Scale_Coordinate_Atomically(int atom_numbers,
                                                          VECTOR *crd) {
  Scale_Vector_Atomically<<<40, 1024>>>(atom_numbers, crd, crd_scale_factor);
}

void MC_BAROSTAT_INFORMATION::Volume_Change_Attempt(VECTOR boxlength) {
  double nrand = ((double)2.0 * rand() / RAND_MAX - 1.0);

  Delta_Box_Length = {0.0f, 0.0f, 0.0f};
  switch (couple_dimension) {
  case NO:
    if (only_direction > 0)
      xyz = only_direction - 1;
    else
      xyz = rand() % 3;
    if (xyz == 0) {
      Delta_Box_Length.x = nrand * Delta_Box_Length_Max[xyz];
    } else if (xyz == 1) {
      Delta_Box_Length.y = nrand * Delta_Box_Length_Max[xyz];
    } else {
      Delta_Box_Length.z = nrand * Delta_Box_Length_Max[xyz];
    }
    break;
  case XY:
    if (only_direction > 0)
      xyz = only_direction - 1;
    else
      xyz = rand() % 2;
    if (xyz == 0) {
      Delta_Box_Length.z = nrand * Delta_Box_Length_Max[xyz];
    } else {
      Delta_Box_Length.y = nrand * Delta_Box_Length_Max[xyz];
      Delta_Box_Length.x = nrand * Delta_Box_Length_Max[xyz];
    }
    break;
  case XZ:
    if (only_direction > 0)
      xyz = only_direction - 1;
    else
      xyz = rand() % 2;
    if (xyz == 0) {
      Delta_Box_Length.y = nrand * Delta_Box_Length_Max[xyz];
    } else {
      Delta_Box_Length.z = nrand * Delta_Box_Length_Max[xyz];
      Delta_Box_Length.x = nrand * Delta_Box_Length_Max[xyz];
    }
    break;
  case YZ:
    if (only_direction > 0)
      xyz = only_direction - 1;
    else
      xyz = rand() % 2;
    if (xyz == 0) {
      Delta_Box_Length.x = nrand * Delta_Box_Length_Max[xyz];
    } else {
      Delta_Box_Length.z = nrand * Delta_Box_Length_Max[xyz];
      Delta_Box_Length.y = nrand * Delta_Box_Length_Max[xyz];
    }
    break;
  case XYZ:
    xyz = 0;
    Delta_Box_Length.x = nrand * Delta_Box_Length_Max[xyz];
    Delta_Box_Length.y = nrand * Delta_Box_Length_Max[xyz];
    Delta_Box_Length.z = nrand * Delta_Box_Length_Max[xyz];
    break;
  }

  New_Box_Length = boxlength + Delta_Box_Length;
  DeltaS = 0.0f;
  switch (couple_dimension) {
  case NO:
    break;
  case XY:
    if (xyz == 1) {
      DeltaS = New_Box_Length.x * New_Box_Length.y - boxlength.x * boxlength.y;
    }
    break;
  case XZ:
    if (xyz == 1) {
      DeltaS = New_Box_Length.x * New_Box_Length.z - boxlength.x * boxlength.z;
    }
    break;
  case YZ:
    if (xyz == 1) {
      DeltaS = New_Box_Length.z * New_Box_Length.y - boxlength.z * boxlength.y;
    }
    break;
  case XYZ:
    break;
  }
  double V = boxlength.x * boxlength.y * boxlength.z;
  newV = New_Box_Length.x * New_Box_Length.y * New_Box_Length.z;
  DeltaV = newV - V;
  VDevided = newV / V;
  crd_scale_factor = New_Box_Length / boxlength;
  // final_term = p0 * DeltaV - N_Beta_Inverse * logf(VDevided);
  // printf("\nDEBUG: %f %f %f\n", crd_scale_factor.x, crd_scale_factor.y,
  // crd_scale_factor.z);
}

int MC_BAROSTAT_INFORMATION::Check_MC_Barostat_Accept() {
  total_count[xyz] += 1;
  if ((float)rand() / RAND_MAX < accept_possibility) {
    reject = 0;
    accep_count[xyz] += 1;
  } else {
    reject = 1;
  }
  return reject;
}

void MC_BAROSTAT_INFORMATION::Initial(CONTROLLER *controller, int atom_numbers,
                                      float target_pressure, VECTOR boxlength,
                                      int res_is_initialized,
                                      const char *module_name) {
  controller->printf("START INITIALIZING MC BAROSTAT:\n");
  if (module_name == NULL) {
    strcpy(this->module_name, "monte_carlo_barostat");
  } else {
    strcpy(this->module_name, module_name);
  }
  controller->printf("    The target pressure is %.2f bar\n",
                     target_pressure * CONSTANT_PRES_CONVERTION);
  V0 = boxlength.x * boxlength.y * boxlength.z;
  newV = V0;
  float mc_baro_initial_ratio = 0.001;
  if (controller[0].Command_Exist(this->module_name, "initial_ratio"))
    mc_baro_initial_ratio =
        atof(controller[0].Command(this->module_name, "initial_ratio"));
  Delta_Box_Length_Max[0] = mc_baro_initial_ratio * boxlength.x;
  Delta_Box_Length_Max[1] = mc_baro_initial_ratio * boxlength.y;
  Delta_Box_Length_Max[2] = mc_baro_initial_ratio * boxlength.z;
  controller->printf("    The initial max box length to change is %f %f %f "
                     "Angstrom for x y z\n",
                     Delta_Box_Length_Max[0], Delta_Box_Length_Max[1],
                     Delta_Box_Length_Max[2]);

  update_interval = 100;
  if (controller[0].Command_Exist(this->module_name, "update_interval"))
    update_interval =
        atoi(controller[0].Command(this->module_name, "update_interval"));
  controller->printf("    The update_interval is %d\n", update_interval);

  check_interval = 10;
  if (controller[0].Command_Exist(this->module_name, "check_interval"))
    check_interval =
        atoi(controller[0].Command(this->module_name, "check_interval"));
  controller->printf("    The check_interval is %d\n", check_interval);

  scale_coordinate_by_molecule = res_is_initialized;
  if (controller[0].Command_Exist(this->module_name, "molecule_scale"))
    scale_coordinate_by_molecule =
        atoi(controller[0].Command(this->module_name, "molecule_scale"));

  controller->printf("    The molecule_scale is %d\n",
                     scale_coordinate_by_molecule);

  accept_rate_low = 30;
  if (controller[0].Command_Exist(this->module_name, "accept_rate_low"))
    accept_rate_low =
        atoi(controller[0].Command(this->module_name, "accept_rate_low"));
  controller->printf("    The lowest accept rate is %.2f%%\n", accept_rate_low);

  accept_rate_high = 40;
  if (controller[0].Command_Exist(this->module_name, "accept_rate_high"))
    accept_rate_high =
        atoi(controller[0].Command(this->module_name, "accept_rate_high"));
  controller->printf("    The highest accept rate is %.2f%%\n",
                     accept_rate_high);

  if (!controller->Command_Exist(this->module_name, "couple_dimension") ||
      controller->Command_Choice(this->module_name, "couple_dimension",
                                 "XYZ")) {
    couple_dimension = XYZ;
  } else if (controller->Command_Choice(this->module_name, "couple_dimension",
                                        "NO")) {
    couple_dimension = NO;
  } else if (controller->Command_Choice(this->module_name, "couple_dimension",
                                        "XY")) {
    couple_dimension = XY;
  } else if (controller->Command_Choice(this->module_name, "couple_dimension",
                                        "XZ")) {
    couple_dimension = XZ;
  } else if (controller->Command_Choice(this->module_name, "couple_dimension",
                                        "YZ")) {
    couple_dimension = YZ;
  }
  if (!controller->Command_Exist(this->module_name, "couple_dimension"))
    controller->printf("    The couple dimension is %s (index %d)\n", "XYZ",
                       couple_dimension);
  else
    controller->printf(
        "    The couple dimension is %s (index %d)\n",
        controller->Command(this->module_name, "couple_dimension"),
        couple_dimension);
  if (controller->Command_Exist(this->module_name, "only_direction")) {
    only_direction =
        atoi(controller->Command(this->module_name, "only_direction"));
  }
  if (couple_dimension != NO && couple_dimension != XYZ) {
    surface_number = 0;
    if (controller->Command_Exist(this->module_name, "surface_number")) {
      surface_number =
          atoi(controller->Command(this->module_name, "surface_number"));
    }
    surface_tension = 0.0f;
    if (controller->Command_Exist(this->module_name, "surface_tension")) {
      surface_tension =
          atof(controller->Command(this->module_name, "surface_tension"));
    }
    surface_tension *= TENSION_UNIT_FACTOR;
    controller->printf("        The surface number is %d\n", surface_number);
    controller->printf("        The surface tension is %f\n", surface_tension);
  }
  Cuda_Malloc_Safely((void **)&frc_backup, sizeof(VECTOR) * atom_numbers);
  Cuda_Malloc_Safely((void **)&crd_backup, sizeof(VECTOR) * atom_numbers);
  is_initialized = 1;
  if (is_initialized && !is_controller_printf_initialized) {
    controller->Step_Print_Initial("density", "%.4f");
    is_controller_printf_initialized = 1;
    controller[0].printf("    structure last modify date is %d\n",
                         last_modify_date);
  }

  controller[0].printf("END INITIALIZING MC BAROSTAT\n\n");
}

void MC_BAROSTAT_INFORMATION::Delta_Box_Length_Max_Update() {
  if (total_count[xyz] % check_interval == 0) {
    accept_rate[xyz] = 100.0 * accep_count[xyz] / total_count[xyz];

    if (accept_rate[xyz] < accept_rate_low) {
      total_count[xyz] = 0;
      accep_count[xyz] = 0;
      Delta_Box_Length_Max[xyz] *= 0.9;
    }
    if (accept_rate[xyz] > accept_rate_high) {
      total_count[xyz] = 0;
      accep_count[xyz] = 0;
      Delta_Box_Length_Max[xyz] *= 1.1;
    }
  }
}

void MC_BAROSTAT_INFORMATION::Ask_For_Calculate_Potential(int steps,
                                                          int *need_potential) {
  if (is_initialized && steps % update_interval == 0) {
    *need_potential = 1;
  }
}
