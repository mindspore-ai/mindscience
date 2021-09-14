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

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/stl_bind.h"

#include "mindelec/ccsrc/api/python/pybind_register.h"
#include "mindelec/ccsrc/scientific_compute/pointcloud/material_analyse.h"
#include "mindelec/ccsrc/scientific_compute/pointcloud/point_cloud_impl.h"
#include "mindelec/ccsrc/scientific_compute/pointcloud/tensor_initializer.h"
#include "mindelec/ccsrc/scientific_compute/pointcloud/utils.h"

namespace minddata {
namespace scientific {
using PhysicalQuantity = pointcloud::PhysicalQuantity;

PYBIND_REGISTER(
  PointCloudImpl, 0, ([](const py::module *m) {
    (void)py::class_<pointcloud::PointCloudImpl, std::shared_ptr<pointcloud::PointCloudImpl>>(*m, "PointCloudImpl")
      .def(py::init([](const std::string &json, const std::string &material_dir, const py::dict &physical_quantity,
                       const int32_t num_of_workers) {
        return std::make_shared<pointcloud::PointCloudImpl>(json, material_dir, physical_quantity, num_of_workers);
      }))
      .def("tensor_init_impl",
           [](pointcloud::PointCloudImpl &impl, const std::vector<double> &sample_x,
            const std::vector<double> &sample_y, const std::vector<double> &sample_z,
            const std::vector<size_t> offset_table, py::array_t<double> *out) {
             THROW_IF_ERROR(impl.TensorInitImpl(sample_x, sample_y, sample_z, offset_table, out));
           })
      .def("material_analyse_impl",
           [](pointcloud::PointCloudImpl &impl, const std::vector<py::dict> &space_solution,
              const std::vector<size_t> &real_model_index, py::array_t<double> *out) {
             THROW_IF_ERROR(impl.MaterialAnalyseImpl(space_solution, real_model_index, out));
           });
  }));

PYBIND_REGISTER(PhysicalQuantity, 0, ([](const py::module *m) {
                  (void)py::enum_<PhysicalQuantity>(*m, "PhysicalQuantity", py::arithmetic())
                    .value("DE_PHYSICAL_MU", PhysicalQuantity::kMu)
                    .value("DE_PHYSICAL_EPSILON", PhysicalQuantity::kEpsilon)
                    .value("DE_PHYSICAL_SIGMA", PhysicalQuantity::kSigma)
                    .value("DE_PHYSICAL_TAND", PhysicalQuantity::kTanD)
                    .export_values();
                }));

}  // namespace scientific
}  // namespace minddata
