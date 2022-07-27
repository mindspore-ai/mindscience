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

#include "Urey_Bradley_force.cuh"

void UREY_BRADLEY::Initial(CONTROLLER *controller, char *module_name) {
  if (module_name == NULL) {
    strcpy(this->module_name, "urey_bradley");
  } else {
    strcpy(this->module_name, module_name);
  }

  char file_name_suffix[CHAR_LENGTH_MAX];
  sprintf(file_name_suffix, "in_file");

  if (controller[0].Command_Exist(this->module_name, file_name_suffix)) {
    controller[0].printf("START INITIALIZING UREY BRADLEY (%s_%s):\n",
                         this->module_name, file_name_suffix);
    FILE *fp = NULL;
    Open_File_Safely(&fp, controller[0].Command(this->module_name, "in_file"),
                     "r");

    int ret = fscanf(fp, "%d", &Urey_Bradley_numbers);
    controller[0].printf("    urey_bradley_numbers is %d\n",
                         Urey_Bradley_numbers);

    bond.bond_numbers = Urey_Bradley_numbers;
    angle.angle_numbers = Urey_Bradley_numbers;

    bond.Memory_Allocate();
    angle.Memory_Allocate();

    for (int i = 0; i < Urey_Bradley_numbers; i++) {
      ret = fscanf(fp, "%d %d %d %f %f %f %f", angle.h_atom_a + i,
                   angle.h_atom_b + i, angle.h_atom_c + i, angle.h_angle_k + i,
                   angle.h_angle_theta0 + i, bond.h_k + i, bond.h_r0 + i);
      bond.h_atom_a[i] = angle.h_atom_a[i];
      bond.h_atom_b[i] = angle.h_atom_c[i];
    }
    fclose(fp);

    bond.Parameter_Host_To_Device();
    angle.Parameter_Host_To_Device();

    bond.is_initialized = 1;
    angle.is_initialized = 1;
    is_initialized = 1;
  } else {
    controller[0].printf("UREY BRADLEY IS NOT INITIALIZED\n\n");
  }

  if (is_initialized && !is_controller_printf_initialized) {
    controller[0].Step_Print_Initial(this->module_name, "%.2f");
    is_controller_printf_initialized = 1;
    controller[0].printf("    structure last modify date is %d\n",
                         last_modify_date);
  }
  if (is_initialized) {
    controller[0].printf("END INITIALIZING UREY BRADLEY\n\n");
  }
}

void UREY_BRADLEY::Urey_Bradley_Force_With_Atom_Energy_And_Virial(
    const UNSIGNED_INT_VECTOR *uint_crd, const VECTOR scaler, VECTOR *frc,
    float *atom_energy, float *atom_virial) {
  if (is_initialized) {
    bond.Bond_Force_With_Atom_Energy_And_Virial(uint_crd, scaler, frc,
                                                atom_energy, atom_virial);
    angle.Angle_Force_With_Atom_Energy(uint_crd, scaler, frc, atom_energy);
  }
}

float UREY_BRADLEY::Get_Energy(const UNSIGNED_INT_VECTOR *uint_crd,
                               const VECTOR scaler, int is_download) {
  if (is_initialized) {
    angle.Get_Energy(uint_crd, scaler, 0);
    bond.Get_Energy(uint_crd, scaler, 0);

    if (is_download) {
      cudaMemcpy(angle.h_sigma_of_angle_ene, angle.d_sigma_of_angle_ene,
                 sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(bond.h_sigma_of_bond_ene, bond.d_sigma_of_bond_ene,
                 sizeof(float), cudaMemcpyDeviceToHost);

      return angle.h_sigma_of_angle_ene[0] + bond.h_sigma_of_bond_ene[0];
    } else {
      return 0;
    }
  }
  return NAN;
}
