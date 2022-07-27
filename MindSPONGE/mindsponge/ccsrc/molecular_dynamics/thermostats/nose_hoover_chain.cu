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

#include "nose_hoover_chain.cuh"

static void Nose_Hoover_Chain_Update(int chain_length, float *chain_crd,
                                     float *chain_vel, float chain_mass,
                                     float Ek, float kB_T, float dt,
                                     int freedom) {
  float chain_mass_inverse = 1.0 / chain_mass;
  chain_vel[0] += (2 * Ek - freedom * kB_T) * chain_mass_inverse * dt -
                  chain_vel[0] * chain_vel[1] * dt;
  for (int i = 1; i < chain_length; i++) {
    float temp_vel = chain_vel[i - 1];
    chain_vel[i] += (temp_vel * temp_vel - kB_T * chain_mass_inverse) * dt -
                    chain_vel[i] * chain_vel[i + 1] * dt;
  }
  for (int i = 0; i < chain_length; i++) {
    chain_crd[i] += chain_vel[i] * dt;
  }
}

static __global__ void MD_Iteration_Leap_Frog_With_NHC(
    const int atom_numbers, const float dt, const float *inverse_mass,
    VECTOR *vel, VECTOR *crd, VECTOR *frc, VECTOR *acc, float chain_vel) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < atom_numbers) {
    acc[i].x = inverse_mass[i] * frc[i].x - vel[i].x * chain_vel;
    acc[i].y = inverse_mass[i] * frc[i].y - vel[i].y * chain_vel;
    acc[i].z = inverse_mass[i] * frc[i].z - vel[i].z * chain_vel;

    vel[i].x = vel[i].x + dt * acc[i].x;
    vel[i].y = vel[i].y + dt * acc[i].y;
    vel[i].z = vel[i].z + dt * acc[i].z;

    crd[i].x = crd[i].x + dt * vel[i].x;
    crd[i].y = crd[i].y + dt * vel[i].y;
    crd[i].z = crd[i].z + dt * vel[i].z;
  }
}

static __global__ void MD_Iteration_Leap_Frog_With_NHC_With_Max_Velocity(
    const int atom_numbers, const float dt, const float *inverse_mass,
    VECTOR *vel, VECTOR *crd, VECTOR *frc, VECTOR *acc, float chain_vel,
    float max_vel) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < atom_numbers) {
    acc[i].x = inverse_mass[i] * frc[i].x - vel[i].x * chain_vel;
    acc[i].y = inverse_mass[i] * frc[i].y - vel[i].y * chain_vel;
    acc[i].z = inverse_mass[i] * frc[i].z - vel[i].z * chain_vel;

    Make_Vector_Not_Exceed_Value(vel[i], max_vel);

    vel[i].x = vel[i].x + dt * acc[i].x;
    vel[i].y = vel[i].y + dt * acc[i].y;
    vel[i].z = vel[i].z + dt * acc[i].z;

    crd[i].x = crd[i].x + dt * vel[i].x;
    crd[i].y = crd[i].y + dt * vel[i].y;
    crd[i].z = crd[i].z + dt * vel[i].z;
  }
}

void NOSE_HOOVER_CHAIN_INFORMATION::Initial(CONTROLLER *controller,
                                            float target_temperature,
                                            const char *module_name) {
  controller->printf("START INITIALIZING NOSE HOOVER CHAIN:\n");
  if (module_name == NULL) {
    strcpy(this->module_name, "nose_hoover_chain");
  } else {
    strcpy(this->module_name, module_name);
  }

  chain_length = 1;
  if (controller[0].Command_Exist(this->module_name, "length")) {
    chain_length = atoi(controller->Command(this->module_name, "length"));
  }
  controller[0].printf("    chain length is %d\n", chain_length);

  Malloc_Safely((void **)&coordinate, sizeof(float) * chain_length);
  Malloc_Safely((void **)&velocity, sizeof(float) * (chain_length + 1));
  Malloc_Safely((void **)&h_mass, sizeof(float) * chain_length);

  float tauT = 1.0f;
  if (controller[0].Command_Exist(this->module_name, "tauT")) {
    tauT = atoi(controller->Command(this->module_name, "tauT"));
  }
  tauT *= CONSTANT_TIME_CONVERTION;
  h_mass = tauT * tauT * target_temperature / 4.0f / CONSTANT_Pi / CONSTANT_Pi;
  kB_T = CONSTANT_kB * target_temperature;
  controller[0].printf("    target temperature is %.2f K\n",
                       target_temperature);
  controller[0].printf("    time constant tau is %f\n", tauT);
  controller[0].printf("    chain mass is %f\n", h_mass);

  if (controller[0].Command_Exist(this->module_name, "restart_input")) {
    FILE *fcrd = NULL;
    Open_File_Safely(
        &fcrd, controller[0].Command(this->module_name, "restart_input"), "r");
    for (int i = 0; i < chain_length; i++) {
      int scan_ret = fscanf(fcrd, "%f %f", coordinate + i, velocity + i);
    }
    fclose(fcrd);
  } else {
    for (int i = 0; i < chain_length; i++) {
      coordinate[i] = 0;
      velocity[i] = 0;
    }
  }
  velocity[chain_length] = 0;

  restart_file_name[0] = 0;
  if (controller[0].Command_Exist(this->module_name, "restart_output")) {
    strcpy(restart_file_name,
           controller->Command(this->module_name, "restart_output"));
  }

  char tempchar[CHAR_LENGTH_MAX];
  tempchar[0] = 0;
  f_crd_traj = NULL;
  if (controller[0].Command_Exist(this->module_name, "crd")) {
    strcpy(tempchar, controller->Command(this->module_name, "crd"));
    Open_File_Safely(&f_crd_traj, tempchar, "w");
  }
  tempchar[0] = 0;
  f_vel_traj = NULL;
  if (controller[0].Command_Exist(this->module_name, "vel")) {
    strcpy(tempchar, controller->Command(this->module_name, "vel"));
    Open_File_Safely(&f_vel_traj, tempchar, "w");
  }

  max_velocity = 0;
  if (controller[0].Command_Exist(this->module_name, "velocity_max")) {
    sscanf(controller[0].Command(this->module_name, "velocity_max"), "%f",
           &max_velocity);
    controller[0].printf("    max velocity is %.2f\n", max_velocity);
  }

  is_initialized = 1;
  if (is_initialized && !is_controller_printf_initialized) {
    is_controller_printf_initialized = 1;
    controller[0].printf("    structure last modify date is %d\n",
                         last_modify_date);
  }

  controller->printf("END INITIALIZING NOSE HOOVER CHAIN\n\n");
}

void NOSE_HOOVER_CHAIN_INFORMATION::MD_Iteration_Leap_Frog(
    int atom_numbers, VECTOR *vel, VECTOR *crd, VECTOR *frc, VECTOR *acc,
    float *inverse_mass, float dt, float Ek, int freedom) {
  if (is_initialized) {
    Nose_Hoover_Chain_Update(chain_length, coordinate, velocity, h_mass, Ek,
                             kB_T, dt, freedom);
    if (max_velocity <= 0) {
      MD_Iteration_Leap_Frog_With_NHC<<<
          (unsigned int)ceilf((float)atom_numbers / 1024), 1024>>>(
          atom_numbers, dt, inverse_mass, vel, crd, frc, acc, velocity[0]);
    } else {
      MD_Iteration_Leap_Frog_With_NHC_With_Max_Velocity<<<
          (unsigned int)ceilf((float)atom_numbers / 1024), 1024>>>(
          atom_numbers, dt, inverse_mass, vel, crd, frc, acc, velocity[0],
          max_velocity);
    }
  }
}

void NOSE_HOOVER_CHAIN_INFORMATION::Save_Restart_File() {
  if (is_initialized && restart_file_name[0] != 0) {
    FILE *frst = NULL;
    Open_File_Safely(&frst, restart_file_name, "w");
    for (int i = 0; i < chain_length; i++) {
      fprintf(frst, "%f %f\n", coordinate[i], velocity[i]);
    }
    fclose(frst);
  }
}

void NOSE_HOOVER_CHAIN_INFORMATION::Save_Trajectory_File() {
  if (is_initialized && f_crd_traj != NULL) {
    for (int i = 0; i < chain_length; i++) {
      fprintf(f_crd_traj, "%f ", coordinate[i]);
    }
    fprintf(f_crd_traj, "\n");
  }
  if (is_initialized && f_vel_traj != NULL) {
    for (int i = 0; i < chain_length; i++) {
      fprintf(f_vel_traj, "%f ", velocity[i]);
    }
    fprintf(f_vel_traj, "\n");
  }
}
