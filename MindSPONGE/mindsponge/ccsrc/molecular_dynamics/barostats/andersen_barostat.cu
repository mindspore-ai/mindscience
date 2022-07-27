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

#include "andersen_barostat.cuh"

void ANDERSEN_BAROSTAT_INFORMATION::Initial(CONTROLLER *controller,
                                            float target_pressure,
                                            VECTOR box_length,
                                            const char *module_name) {
  controller->printf("START INITIALIZING ANDERSEN BAROSTAT:\n");
  if (module_name == NULL) {
    strcpy(this->module_name, "andersen_barostat");
  } else {
    strcpy(this->module_name, module_name);
  }
  controller->printf("    The target pressure is %.2f bar\n",
                     target_pressure * CONSTANT_PRES_CONVERTION);

  V0 = box_length.x * box_length.y * box_length.z;
  new_V = V0;

  float taup = 1.0f;
  if (controller[0].Command_Exist(this->module_name, "tau"))
    taup = atof(controller[0].Command(this->module_name, "tau"));
  controller->printf("    The time constant tau is %f ps\n", taup);

  float compressibility = 4.5e-5f;
  if (controller[0].Command_Exist(this->module_name, "compressibility"))
    compressibility =
        atof(controller[0].Command(this->module_name, "compressibility"));
  controller->printf("    The compressibility constant is %f bar^-1\n",
                     compressibility);

  h_mass_inverse = taup * taup / V0 / compressibility;
  controller->printf("    The piston mass is %f bar·ps^2·A^-3\n",
                     h_mass_inverse);

  taup *= CONSTANT_TIME_CONVERTION;
  compressibility *= CONSTANT_PRES_CONVERTION;
  h_mass_inverse = V0 * compressibility / taup / taup;

  dV_dt = 0;
  if (controller[0].Command_Exist(this->module_name, "dV/dt"))
    dV_dt = atof(controller[0].Command(this->module_name, "dV/dt"));
  controller->printf("    The initial dV/dt is %f A^3/(20.455 fs)\n", dV_dt);

  is_initialized = 1;
  if (is_initialized && !is_controller_printf_initialized) {
    controller->Step_Print_Initial("density", "%.4f");
    controller->Step_Print_Initial("pressure", "%.2f");
    controller->Step_Print_Initial("dV/dt", "%.2f");
    is_controller_printf_initialized = 1;
    controller[0].printf("    structure last modify date is %d\n",
                         last_modify_date);
  }

  controller->printf("END INITIALIZING BERENDSEN BAROSTAT\n\n");
}
