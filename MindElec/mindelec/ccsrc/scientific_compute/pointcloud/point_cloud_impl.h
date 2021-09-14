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

#ifndef MINDELEC_CCSRC_SCIENTIFIC_COMPUTE_POINTCLOUD_POINT_CLOUD_IMPL_H_
#define MINDELEC_CCSRC_SCIENTIFIC_COMPUTE_POINTCLOUD_POINT_CLOUD_IMPL_H_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <memory>
#include <string>
#include <vector>

#include "mindelec/ccsrc/scientific_compute/pointcloud/tensor_initializer.h"

namespace minddata {
namespace scientific {
namespace pointcloud {

namespace py = pybind11;

class MaterialAnalyser;

class PointCloudImpl {
 public:
  // Constructor
  // @param std::string json - json file path
  // @param std::string material_dir - material files directory path
  // @param py::dict physical_quantity - physical quantity default value dictionary
  // @param int32_t num_of_workers - parallel arguments
  PointCloudImpl(const std::string &json, const std::string &material_dir, const py::dict &physical_quantity,
                 const int32_t num_of_workers);

  // Default constructor
  PointCloudImpl();

  ~PointCloudImpl();

  // API for tensor initialization
  // @param std::vector<double> sample_x - x_sampling space
  // @param std::vector<double> sample_y - y_sampling space
  // @param std::vector<double> sample_z - z_sampling space
  // @param std::vector<size_t> offset_table - offset table for pointcloud tensor
  // @param py::array_t<double> *out - Initialized result pointcloud tensor
  // @return Status - the status code returned
  Status TensorInitImpl(const std::vector<double> &sample_x, const std::vector<double> &sample_y,
                        const std::vector<double> &sample_z, const std::vector<size_t> offset_table,
                        py::array_t<double> *out) const;

  // API for material analysis
  // @param std::vector<py::dict> space_solution - space solution obtained in space solving module at python layer
  // @param std::vector<size_t> real_model_index - real model index for each sub-model
  // @param py::array_t<double> *out - Pointcloud tensor with material information
  // @return Status - the status code returned
  Status MaterialAnalyseImpl(const std::vector<py::dict> &space_solution, const std::vector<size_t> &real_model_index,
                             py::array_t<double> *out) const;

 private:
  std::shared_ptr<MaterialAnalyser> material_solver_ptr_;
  std::shared_ptr<TensorInitializer> initializer_ptr_;
};


}  // namespace pointcloud
}  // namespace scientific
}  // namespace minddata

#endif  // MINDELEC_CCSRC_SCIENTIFIC_COMPUTE_POINTCLOUD_POINT_CLOUD_IMPL_H_
