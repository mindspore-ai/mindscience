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

namespace minddata {
namespace scientific {
namespace pointcloud {

MaterialAnalyser::MaterialAnalyser(const std::string &json, const std::string &material_dir,
                                   const int32_t num_of_workers, const py::dict &physical_quantity)
  : json_(json), material_dir_(material_dir), num_of_workers_(num_of_workers) {
  for (auto &item : physical_quantity) {
    auto field = item.first.cast<PhysicalQuantity>();
    auto default_value = item.second.cast<double>();

    if (field == PhysicalQuantity::kMu) {
      special_quantity_information_.emplace_back(std::make_tuple(".Mue", default_value));
    }

    auto iter = physical_name2string_map.find(field);
    physical_quantity_information_.emplace_back(std::make_tuple(iter->second, default_value));
  }
  if (material_dir_.back() != '/') {
    material_dir_ += '/';
  }

  LOG(INFO) << physical_quantity_information_.size() << " physical quantities to process totally";
}

MaterialAnalyser::~MaterialAnalyser() = default;

Status MaterialAnalyser::MaterialAnalyseImpl(const std::vector<py::dict> &space_solution,
                                             const std::vector<size_t> &real_model_index,
                                             py::array_t<double> *out) const {
  auto begin = std::chrono::steady_clock::now();

  std::ifstream ifs(json_.c_str(), std::ios::in);
  CHECK_FAIL_RETURN_UNEXPECTED(ifs.good(), "Material json file: " + json_ + " is not valid");
  nlohmann::json material_dict;
  ifs >> material_dict;

  std::unordered_map<size_t, std::vector<double>> material_cache;

  for (auto &item : material_dict["solids"]) {
    std::string material_name = item["material"];
    size_t model_idx = item["index"];
    std::vector<double> physical_info;
    RETURN_IF_NOT_OK(PerformMaterialValue_(material_name, &physical_info));
    material_cache.insert(std::make_pair(model_idx, physical_info));
  }

  py::buffer_info info = out->request();
  double *data_ptr = static_cast<double *>(info.ptr);
  LOG(INFO) << "Total space solution size: " << space_solution.size();

  auto real_thread_num = std::min(static_cast<unsigned int>(num_of_workers_), std::thread::hardware_concurrency());
  LOG(INFO) << real_thread_num << " threads will be applied";

  size_t block_size = space_solution.size() / real_thread_num;
  std::vector<std::future<Status>> thread_exc_rc(real_thread_num - 1);
  auto block_start = space_solution.begin();
  for (size_t t_id = 0; t_id < real_thread_num - 1; t_id++) {
    auto block_end = block_start;
    std::advance(block_end, block_size);
    thread_exc_rc[t_id] = std::async(std::launch::async, &MaterialAnalyser::AssignmentTensor_, this, block_start,
                                      block_end, std::ref(material_cache), std::ref(real_model_index), data_ptr);
    block_start = block_end;
  }

  RETURN_IF_NOT_OK(AssignmentTensor_(block_start, space_solution.end(), material_cache, real_model_index, data_ptr));

  for (size_t t_id = 0; t_id < thread_exc_rc.size(); t_id++) {
    CHECK_FAIL_RETURN_UNEXPECTED(thread_exc_rc[t_id].get().IsOk(), "Thread " + std::to_string(t_id) +
    " execution fail.");
  }

  auto end = std::chrono::steady_clock::now();
  LOG(INFO) << "Material analysis finished successfully. Time costs: "
               << std::chrono::duration<double, std::milli>(end - begin).count() << " ms";
  return Status::OK();
}

Status MaterialAnalyser::PerformMaterialValue_(const std::string &material_name, std::vector<double> *out) const {
  auto index = material_name.find_last_of('/');
  auto real_material_name = material_name.substr(index + 1, std::string::npos);
  std::string target_file_name = real_material_name + ".mtd";

  size_t phy_quantity_num = physical_quantity_information_.size();

  out->reserve(phy_quantity_num);
  for (auto &phy_info : physical_quantity_information_) {
    out->emplace_back(std::get<1>(phy_info));
  }

  std::vector<std::tuple<std::string, double>> real_physical_quantity_info = physical_quantity_information_;

  if (!special_quantity_information_.empty()) {
    for (auto &name : special_quantity_information_) {
      real_physical_quantity_info.emplace_back(name);
    }
  }

  std::ifstream infile;
  infile.open(material_dir_ + target_file_name, std::ios::in);
  std::string error_msg = "File path: " + material_dir_ + target_file_name + " is invalid";
  CHECK_FAIL_RETURN_UNEXPECTED(infile.is_open(), error_msg);

  std::string one_line;
  uint8_t trig = 0;
  std::string scientific{"[+-]?[0-9]+(.[0-9]+(e[0-9]+)?)?"};
  std::string strip_string = " ";
  while (std::getline(infile, one_line)) {
    RETURN_IF_NOT_OK(Strip(strip_string, &one_line));

    size_t quantity_index = 0;
    for (auto &phy_information : real_physical_quantity_info) {
      std::string phy_name = std::get<0>(phy_information);
      if (IsOnlyStartWith(one_line, phy_name)) {
        std::regex re(scientific);
        std::smatch match;

        trig++;
        bool ret = std::regex_search(one_line, match, re);
        if (ret) {
          size_t real_quantity_index = quantity_index % phy_quantity_num;
          (*out)[real_quantity_index] = std::stod(match[0].str());
        }
        break;
      }
      quantity_index++;
    }
    if (trig == phy_quantity_num) {
      break;
    }
  }
  return Status::OK();
}

Status MaterialAnalyser::AssignmentTensor_(std::vector<py::dict>::const_iterator block_begin,
                                           std::vector<py::dict>::const_iterator block_end,
                                           const std::unordered_map<size_t, std::vector<double>> &material_cache,
                                           const std::vector<size_t> &real_model_index, double *out) const {
  auto thread_task_number = std::distance(block_begin, block_end);
  for (auto iter = block_begin; iter != block_end; iter++) {
    for (auto &item : (*iter)) {
      auto offset_base = item.first.cast<size_t>();
      auto model_column_offset = offset_base + StartColumn;
      size_t real_model_id = real_model_index[item.second.cast<size_t>()];

      if (real_model_id > out[model_column_offset]) {
        out[model_column_offset] = real_model_id;
        auto material_iter = material_cache.find(real_model_id);

        CHECK_FAIL_RETURN_UNEXPECTED(material_iter != material_cache.end(), "Current model id: "
                                     + std::to_string(real_model_id) + " not exist in json file.");

        std::vector<double> physical_info = material_iter->second;
        for (size_t i = 0; i < physical_info.size(); i++) {
          out[model_column_offset + i + 1] = physical_info[i];
        }
      }
    }
  }
  LOG(INFO) << thread_task_number << " solutions have been processed";
  return Status::OK();
}

}  // namespace pointcloud
}  // namespace scientific
}  // namespace minddata
