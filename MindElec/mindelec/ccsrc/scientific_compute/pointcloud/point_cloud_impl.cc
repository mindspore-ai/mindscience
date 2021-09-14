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

#include "mindelec/ccsrc/scientific_compute/pointcloud/material_analyse.h"
#include "mindelec/ccsrc/scientific_compute/pointcloud/point_cloud_impl.h"

namespace minddata {
namespace scientific {
namespace pointcloud {

PointCloudImpl::PointCloudImpl(const std::string &json, const std::string &material_dir,
                               const py::dict &physical_quantity, const int32_t num_of_workers)
  : material_solver_ptr_(std::make_shared<MaterialAnalyser>(json, material_dir, num_of_workers, physical_quantity)),
    initializer_ptr_(std::make_shared<TensorInitializer>(physical_quantity)) {
  LOG(INFO) << "Construct with material information";
}

PointCloudImpl::PointCloudImpl() : material_solver_ptr_(nullptr), initializer_ptr_(nullptr) {
  LOG(WARNING) << "Construct without material information, thus all material analyse functions will not be supported";
}

PointCloudImpl::~PointCloudImpl() = default;

Status PointCloudImpl::TensorInitImpl(const std::vector<double> &sample_x, const std::vector<double> &sample_y,
                                      const std::vector<double> &sample_z, const std::vector<size_t> offset_table,
                                      py::array_t<double> *out) const {
  CHECK_FAIL_RETURN_UNEXPECTED(initializer_ptr_, "Construct without material config, TensorInitImpl is not supported "
                                                 "under this situation");
  RETURN_IF_NOT_OK(initializer_ptr_->TensorInitImpl(sample_x, sample_y, sample_z, offset_table, out));
  return Status::OK();
}

Status PointCloudImpl::MaterialAnalyseImpl(const std::vector<py::dict> &space_solution,
                                           const std::vector<size_t> &real_model_index,
                                           py::array_t<double> *out) const {
  CHECK_FAIL_RETURN_UNEXPECTED(material_solver_ptr_, "Construct without material config, MaterialAnalyseImpl is not "
                                                     "supported under this situation.");
  RETURN_IF_NOT_OK(material_solver_ptr_->MaterialAnalyseImpl(space_solution, real_model_index, out));
  return Status::OK();
}

}  // namespace pointcloud
}  // namespace scientific
}  // namespace minddata
