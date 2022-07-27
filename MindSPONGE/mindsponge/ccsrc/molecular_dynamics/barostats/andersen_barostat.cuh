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

#ifndef ANDERSEN_BAROSTAT_CUH
#define ANDERSEN_BAROSTAT_CUH
#include "../common.cuh"
#include "../control.cuh"

//用于记录与计算Andersen控压相关的信息
struct ANDERSEN_BAROSTAT_INFORMATION {
  char module_name[CHAR_LENGTH_MAX];
  int is_initialized = 0;
  int is_controller_printf_initialized = 0;
  int last_modify_date = 20211029;

  double dV_dt = 0; //拓展自由度的速度
  double V0, new_V; //初始体积和新体积
  double crd_scale_factor = 1;
  float h_mass_inverse; //拓展自由度的质量的倒数

  //初始化
  void Initial(CONTROLLER *controller, float target_pressure, VECTOR box_length,
               const char *module_name = NULL);
};

#endif // ANDERSEN_THERMOSTAT_CUH(Anderson.cuh)
