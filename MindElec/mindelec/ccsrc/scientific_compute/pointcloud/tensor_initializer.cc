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


#include "mindelec/ccsrc/scientific_compute/pointcloud/tensor_initializer.h"

namespace minddata {
namespace scientific {
namespace pointcloud {

TensorInitializer::TensorInitializer(const py::dict &physical_quantity) {
  size_t property_column_id = 3;
  real_column_id_.emplace_back(std::make_tuple(property_column_id, ModelDefaultValue));

  for (auto &item : physical_quantity) {
    property_column_id++;

    auto field = item.first.cast<PhysicalQuantity>();
    double default_value = item.second.cast<double>();

    auto iter = physical_name2string_map.find(field);
    phy_default_value_.insert(std::make_pair(iter->second, default_value));
    if (default_value != 0.) {
      real_column_id_.emplace_back(std::make_tuple(property_column_id, default_value));
    }
  }
  LOG(INFO) << phy_default_value_.size() << " physical quantities to process totally";
}

TensorInitializer::~TensorInitializer() = default;

Status TensorInitializer::TensorInitImpl(const std::vector<double> &sample_x, const std::vector<double> &sample_y,
                                         const std::vector<double> &sample_z, const std::vector<size_t> offset_table,
                                         py::array_t<double> *out) const {
  auto begin = std::chrono::steady_clock::now();
  CHECK_FAIL_RETURN_UNEXPECTED(!sample_x.empty(), "Sampling_X can not be empty vector");
  CHECK_FAIL_RETURN_UNEXPECTED(!sample_y.empty(), "Sampling_Y can not be empty vector");
  CHECK_FAIL_RETURN_UNEXPECTED(!sample_z.empty(), "Sampling_Z can not be empty vector");
  CHECK_FAIL_RETURN_UNEXPECTED(offset_table.size() == NDim, "Offset table size must be 4");
  CHECK_FAIL_RETURN_UNEXPECTED(out != nullptr, "nd-array can not be empty pointer");

  constexpr size_t size_init_value = 1;
  py::buffer_info info = out->request();
  double *data_ptr = static_cast<double *>(info.ptr);
  size_t total_size = std::accumulate(info.shape.begin(), info.shape.end(), size_init_value, std::multiplies<size_t>());

  LOG(INFO) << "Final tensor total size: " << total_size;

  size_t need_process_column = phy_default_value_.size() + NDim;
  for (auto &item : phy_default_value_) {
    if (item.second == TensorDefaultValue) {
      need_process_column -= 1;
    }
  }

  unsigned int thread_num = 2 * need_process_column;

  std::vector<std::future<Status>> thread_exc_rc(thread_num);
  size_t half_number = need_process_column;
  LOG(INFO) << "Exist " << half_number << " columns to be processed";

  for (size_t column = 0; column < half_number; column++) {
    if (column == 0) {
      thread_exc_rc[column] = std::async(std::launch::async, &TensorInitializer::ColumnX_, this, std::ref(sample_x),
                                         std::ref(offset_table), HalfSegment::kFirstHalf, data_ptr);
      thread_exc_rc[column + half_number] = std::async(std::launch::async, &TensorInitializer::ColumnX_, this,
                                                       std::ref(sample_x), std::ref(offset_table),
                                                       HalfSegment::kSecondHalf, data_ptr);
    } else if (column == 1) {
      thread_exc_rc[column] = std::async(std::launch::async, &TensorInitializer::ColumnY_, this, std::ref(sample_y),
                                         std::ref(offset_table), HalfSegment::kFirstHalf, sample_x.size(), data_ptr);
      thread_exc_rc[column + half_number] = std::async(std::launch::async, &TensorInitializer::ColumnY_, this,
                                                       std::ref(sample_y), std::ref(offset_table),
                                                       HalfSegment::kSecondHalf, sample_x.size(), data_ptr);
    } else if (column == 2) {
      thread_exc_rc[column] = std::async(std::launch::async, &TensorInitializer::ColumnZ_, this, std::ref(sample_z),
                                         std::ref(offset_table), HalfSegment::kFirstHalf, total_size, data_ptr);
      thread_exc_rc[column + half_number] = std::async(std::launch::async, &TensorInitializer::ColumnZ_, this,
                                                       std::ref(sample_z), std::ref(offset_table),
                                                       HalfSegment::kSecondHalf, total_size, data_ptr);
    } else {
      size_t real_column = std::get<0>(real_column_id_[column - 3]);
      double default_value = std::get<1>(real_column_id_[column - 3]);
      thread_exc_rc[column] = std::async(std::launch::async, &TensorInitializer::ColumnModel_, this, std::ref(sample_x),
                                         std::ref(offset_table), HalfSegment::kFirstHalf, default_value, real_column,
                                         data_ptr);
      thread_exc_rc[column + half_number] = std::async(std::launch::async, &TensorInitializer::ColumnModel_, this,
                                                       std::ref(sample_x), std::ref(offset_table),
                                                       HalfSegment::kSecondHalf, default_value, real_column, data_ptr);
    }
  }
  for (size_t i = 0; i < thread_exc_rc.size(); i++) {
    CHECK_FAIL_RETURN_UNEXPECTED(thread_exc_rc[i].get().IsOk(), "Thread " + std::to_string(i) + " execute task fail.");
  }
  auto end = std::chrono::steady_clock::now();
  LOG(INFO) << "Tensor initialize successfully. Time costs: "
              << std::chrono::duration<double, std::milli>(end - begin).count() << " ms";
  return Status::OK();
}

Status TensorInitializer::ColumnX_(const std::vector<double> &sample_x, const std::vector<size_t> offset_table,
                                   const HalfSegment segment_flag, double *numpy_ptr) const {
  const size_t column_index = 0;
  const size_t unit_offset = offset_table[2];
  std::string info_out = (segment_flag == HalfSegment::kFirstHalf) ? "first" : "second";
  LOG(INFO) << "Column X at " << info_out << " segment begin at thread number: " << std::this_thread::get_id();

  size_t seg_start = (segment_flag == HalfSegment::kFirstHalf) ? 0 : sample_x.size() / 2;
  size_t seg_end = (segment_flag == HalfSegment::kFirstHalf) ? sample_x.size() / 2 : sample_x.size();

  for (size_t idx = seg_start; idx < seg_end; idx++) {
    double value = sample_x[idx];
    size_t cur_idx_begin = offset_table[0] * idx;
    size_t cur_idx_end = offset_table[0] * (idx + 1);
    for (size_t offset = cur_idx_begin; offset < cur_idx_end; offset += unit_offset) {
      numpy_ptr[offset + column_index] = value;
    }
  }
  return Status::OK();
}

Status TensorInitializer::ColumnY_(const std::vector<double> &sample_y, const std::vector<size_t> offset_table,
                                   const HalfSegment segment_flag, const size_t size_x, double *numpy_ptr) const {
  const size_t column_index = 1;
  const size_t unit_offset = offset_table[2];
  std::string info_out = (segment_flag == HalfSegment::kFirstHalf) ? "first" : "second";
  LOG(INFO) << "Column Y at " << info_out << " segment begin at thread number: " << std::this_thread::get_id();

  size_t seg_start = (segment_flag == HalfSegment::kFirstHalf) ? 0 : sample_y.size() / 2;
  size_t seg_end = (segment_flag == HalfSegment::kFirstHalf) ? sample_y.size() / 2 : sample_y.size();

  for (size_t idy = seg_start; idy < seg_end; idy++) {
    double value = sample_y[idy];
    for (size_t idx = 0; idx < size_x; idx++) {
      size_t cur_idx_begin = idx * offset_table[0] + idy * offset_table[1];
      size_t cur_idx_end = idx * offset_table[0] + (idy + 1) * offset_table[1];
      for (size_t offset = cur_idx_begin; offset < cur_idx_end; offset += unit_offset) {
        numpy_ptr[offset + column_index] = value;
      }
    }
  }
  return Status::OK();
}

Status TensorInitializer::ColumnZ_(const std::vector<double> &sample_z, const std::vector<size_t> offset_table,
                                   const HalfSegment segment_flag, const size_t total_size, double *numpy_ptr) const {
  const size_t column_index = 2;
  const size_t unit_offset = offset_table[1];
  std::string info_out = (segment_flag == HalfSegment::kFirstHalf) ? "first" : "second";
  LOG(INFO) << "Column Z at " << info_out << " segment begin at thread number: " << std::this_thread::get_id();

  size_t seg_start = (segment_flag == HalfSegment::kFirstHalf) ? 0 : sample_z.size() / 2;
  size_t seg_end = (segment_flag == HalfSegment::kFirstHalf) ? sample_z.size() / 2 : sample_z.size();

  for (size_t idz = seg_start; idz < seg_end; idz++) {
    double value = sample_z[idz];
    size_t cur_idz_begin = idz * offset_table[2];
    size_t cur_idz_end = total_size + (idz - sample_z.size() + 1) * offset_table[2];
    for (size_t offset = cur_idz_begin; offset < cur_idz_end; offset += unit_offset) {
      numpy_ptr[offset + column_index] = value;
    }
  }
  return Status::OK();
}

Status TensorInitializer::ColumnModel_(const std::vector<double> &sample_x, const std::vector<size_t> offset_table,
                                       const HalfSegment segment_flag, const double default_value,
                                       const size_t column_index, double *numpy_ptr) const {
  const size_t unit_offset = offset_table[2];
  std::string info_out = (segment_flag == HalfSegment::kFirstHalf) ? "first" : "second";
  LOG(INFO) << "Column " << column_index << " at " << info_out << " segment begin at thread number: "
              << std::this_thread::get_id();

  size_t seg_start = (segment_flag == HalfSegment::kFirstHalf) ? 0 : sample_x.size() / 2;
  size_t seg_end = (segment_flag == HalfSegment::kFirstHalf) ? sample_x.size() / 2 : sample_x.size();

  size_t offset_begin = seg_start * offset_table[0];
  size_t offset_end = seg_end * offset_table[0];
  for (size_t offset = offset_begin; offset < offset_end; offset += unit_offset) {
    numpy_ptr[offset + column_index] = default_value;
  }
  return Status::OK();
}

}  // namespace pointcloud
}  // namespace scientific
}  // namespace minddata
