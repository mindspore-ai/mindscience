/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MINDELEC_CCSRC_SCIENTIFIC_COMPUTE_POINTCLOUD_MATERIAL_ANALYSE_H_
#define MINDELEC_CCSRC_SCIENTIFIC_COMPUTE_POINTCLOUD_MATERIAL_ANALYSE_H_
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <algorithm>
#include <iterator>
#include <fstream>
#include <future>
#include <regex>
#include <thread>
#include <utility>
#include <vector>
#include <tuple>
#include <string>
#include <unordered_map>
#include <nlohmann/json.hpp>

#include "mindelec/ccsrc/core/utils/log_adapter.h"
#include "mindelec/ccsrc/scientific_compute/pointcloud/utils.h"
#include "mindelec/ccsrc/include/status.h"

namespace py = pybind11;

namespace minddata {
namespace scientific {
namespace pointcloud {

constexpr size_t StartColumn = 3;

class MaterialAnalyser {
 public:
  // Constructor
  // @param std::string json - json file path
  // @param std::string material_dir - material files directory path
  // @param int32_t num_of_workers - parallel arguments
  // @param py::dict physical_quantity - physical quantity default value dictionary
  MaterialAnalyser(const std::string &json, const std::string &material_dir, const int32_t num_of_workers,
                   const py::dict &physical_quantity);

  ~MaterialAnalyser();

  // Implementation for material analysis
  // @param std::vector<py::dict> space_solution - space solution obtained in space solving module at python layer
  // @param std::vector<size_t> real_model_index - real model index for each sub-model
  // @param py::array_t<double> *out - Pointcloud tensor with material information
  // @return Status - the status code returned
  Status MaterialAnalyseImpl(const std::vector<py::dict> &space_solution, const std::vector<size_t> &real_model_index,
                             py::array_t<double> *out) const;

 private:
  std::string json_;
  std::string material_dir_;
  int32_t num_of_workers_;
  std::vector<std::tuple<std::string, double>> physical_quantity_information_;
  std::vector<std::tuple<std::string, double>> special_quantity_information_;

  Status PerformMaterialValue_(const std::string &material_name, std::vector<double> *out) const;

  Status AssignmentTensor_(std::vector<py::dict>::const_iterator block_begin,
                           std::vector<py::dict>::const_iterator block_end,
                           const std::unordered_map<size_t, std::vector<double>> &material_cache,
                           const std::vector<size_t> &real_model_index, double *out) const;
};

}  // namespace pointcloud
}  // namespace scientific
}  // namespace minddata

#endif  // MINDELEC_CCSRC_SCIENTIFIC_COMPUTE_POINTCLOUD_MATERIAL_ANALYSE_H_
