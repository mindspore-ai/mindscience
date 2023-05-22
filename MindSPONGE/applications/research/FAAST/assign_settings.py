# Copyright 2023 Huawei Technologies Co., Ltd & CPL YiQin GAO Research Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"assign_settings"
import copy

settings = {}
settings["calibration"] = {'volume_or_intensity': 'volume',
                           'relaxation_matrix': 'no',
                           'distance_cutoff': 6.0,
                           'estimator': 'ratio_of_averages',
                           'error_estimator': 'distance',
                           'use_bounds': 'no'}
settings["violation_analysis"] = {'violation_tolerance': 1.0,
                                  'lower_bound_correction': {
                                      "enabled": "no",
                                      "value": 0.0,
                                  },
                                  'upper_bound_correction': {
                                      "enabled": "no",
                                      "value": 0.0,
                                  },
                                  'violation_threshold': 0.5,
                                  'sigma_mode': 'fix'}
settings["assign"] = {'max_contributions': 20,
                      'weight_cutoff': 0.9}
settings["infer_pdb"] = {'sample_ur_rate': 0.05,
                         "num_repeats": 20}

assign_all_settings = {}

settings.get("infer_pdb")["sample_ur_rate"] = 0.0
settings.get("infer_pdb")["num_repeats"] = 1
settings.get("assign")["weight_cutoff"] = 0.9
settings["init_assign"] = True
assign_all_settings[0] = copy.deepcopy(settings)

settings.get("infer_pdb")["sample_ur_rate"] = 0.10
settings.get("infer_pdb")["num_repeats"] = 20
settings.get("assign")["weight_cutoff"] = 0.9
settings["init_assign"] = False
assign_all_settings[1] = copy.deepcopy(settings)

settings.get("infer_pdb")["sample_ur_rate"] = 0.20
settings.get("infer_pdb")["num_repeats"] = 20
settings.get("assign")["weight_cutoff"] = 0.8
settings["init_assign"] = False
assign_all_settings[2] = copy.deepcopy(settings)
