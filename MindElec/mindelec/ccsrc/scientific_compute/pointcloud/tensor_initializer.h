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

#ifndef MINDELEC_CCSRC_SCIENTIFIC_COMPUTE_POINTCLOUD_TENSOR_INIT_H_
#define MINDELEC_CCSRC_SCIENTIFIC_COMPUTE_POINTCLOUD_TENSOR_INIT_H_

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <chrono>
#include <functional>
#include <future>
#include <numeric>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
#include <algorithm>

#include "mindelec/ccsrc/core/utils/log_adapter.h"
#include "mindelec/ccsrc/scientific_compute/pointcloud/utils.h"
#include "mindelec/ccsrc/include/status.h"


namespace py = pybind11;

namespace minddata {
namespace scientific {
namespace pointcloud {

constexpr double TensorDefaultValue = 0.;
constexpr size_t NDim = 4;
constexpr double ModelDefaultValue = -1.;


enum class HalfSegment {
  kFirstHalf = 0,
  kSecondHalf = 1
};

class TensorInitializer {
 public:
  // Constructor
  // @param py::dict physical_quantity - physical quantity default value dictionary
  explicit TensorInitializer(const py::dict &physical_quantity);

  ~TensorInitializer();

  // Implementation for tensor initialization
  // @param std::vector<double> sample_x - x_sampling space
  // @param std::vector<double> sample_y - y_sampling space
  // @param std::vector<double> sample_z - z_sampling space
  // @param std::vector<size_t> offset_table - offset table for pointcloud tensor
  // @param py::array_t<double> *out - Initialized result pointcloud tensor
  // @return Status - the status code returned
  Status TensorInitImpl(const std::vector<double> &sample_x, const std::vector<double> &sample_y,
                        const std::vector<double> &sample_z, const std::vector<size_t> offset_table,
                        py::array_t<double> *out) const;

 private:
  std::unordered_map<std::string, double> phy_default_value_;
  std::vector<std::tuple<size_t, double>> real_column_id_;

  // Task column x, define the task to be done by one thread
  // @param std::vector<double> sample_x - x_sampling space
  // @param std::vector<size_t> offset_table - offset base table for computing offse value
  // @param HalfSegment segment_flag - first or second half
  // @param double *numpy_ptr - Final PointCloud tensor embedded data pointer
  // @return Status - the status code returned
  Status ColumnX_(const std::vector<double> &sample_x, const std::vector<size_t> offset_table,
                  const HalfSegment segment_flag, double *numpy_ptr) const;

  // Task column y, define the task to be done by one thread
  // @param std::vector<double> sample_y - y_sampling space
  // @param std::vector<size_t> offset_table - offset base table for computing offse value
  // @param HalfSegment segment_flag - first or second half
  // @param size_t size_x - x_sampling size
  // @param double *numpy_ptr - Final PointCloud tensor embedded data pointer
  // @return Status - the status code returned
  Status ColumnY_(const std::vector<double> &sample_y, const std::vector<size_t> offset_table,
                  const HalfSegment segment_flag, const size_t size_x, double *numpy_ptr) const;

  // Task column z, define the task to be done by one thread
  // @param std::vector<double> sample_z - z_sampling space
  // @param std::vector<size_t> offset_table - offset base table for computing offse value
  // @param HalfSegment segment_flag - first or second half
  // @param size_t total_size - PointCloud tensor total data size
  // @param double *numpy_ptr - Final PointCloud tensor embedded data pointer
  // @return Status - the status code returned
  Status ColumnZ_(const std::vector<double> &sample_z, const std::vector<size_t> offset_table,
                  const HalfSegment segment_flag, const size_t total_size, double *numpy_ptr) const;

  // Task column others, define the task to be done by one thread
  // @param std::vector<double> sample_x - x_sampling space
  // @param std::vector<size_t> offset_table - offset base table for computing offse value
  // @param HalfSegment segment_flag - first or second half
  // @param default_value - default value for current column
  // @param size_t column_index - column_index that determine the final offset
  // @param double *numpy_ptr - Final PointCloud tensor embedded data pointer
  // @return Status - the status code returned
  Status ColumnModel_(const std::vector<double> &sample_x, const std::vector<size_t> offset_table,
                      const HalfSegment segment_flag, const double default_value, const size_t column_index,
                      double *numpy_ptr) const;
};


}  // namespace pointcloud
}  // namespace scientific
}  // namespace minddata

#endif  // MINDELEC_CCSRC_SCIENTIFIC_COMPUTE_POINTCLOUD_TENSOR_INIT_H_
