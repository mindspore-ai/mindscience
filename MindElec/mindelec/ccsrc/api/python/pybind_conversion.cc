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

#include "mindelec/ccsrc/core/utils/log_adapter.h"
#include "mindelec/ccsrc/api/python/pybind_conversion.h"

namespace minddata {

float toFloat(const py::handle &handle) { return py::reinterpret_borrow<py::float_>(handle); }

int toInt(const py::handle &handle) { return py::reinterpret_borrow<py::int_>(handle); }

int64_t toInt64(const py::handle &handle) { return py::reinterpret_borrow<py::int_>(handle); }

bool toBool(const py::handle &handle) { return py::reinterpret_borrow<py::bool_>(handle); }

std::string toString(const py::handle &handle) { return py::reinterpret_borrow<py::str>(handle); }

std::set<std::string> toStringSet(const py::list list) {
  std::set<std::string> set;
  if (!list.empty()) {
    for (auto l : list) {
      if (!l.is_none()) {
        (void)set.insert(py::str(l));
      }
    }
  }
  return set;
}

std::map<std::string, int32_t> toStringMap(const py::dict dict) {
  std::map<std::string, int32_t> map;
  if (!dict.empty()) {
    for (auto p : dict) {
      (void)map.emplace(toString(p.first), toInt(p.second));
    }
  }
  return map;
}

std::vector<std::string> toStringVector(const py::list list) {
  std::vector<std::string> vector;
  if (!list.empty()) {
    for (auto l : list) {
      if (l.is_none())
        vector.emplace_back("");
      else
        vector.push_back(py::str(l));
    }
  }
  return vector;
}

std::vector<pid_t> toIntVector(const py::list input_list) {
  std::vector<pid_t> vector;
  if (!input_list.empty()) {
    std::transform(input_list.begin(), input_list.end(), std::back_inserter(vector),
                   [&](const py::handle &handle) { return static_cast<pid_t>(toInt(handle)); });
  }
  return vector;
}

std::unordered_map<int32_t, std::vector<pid_t>> toIntMap(const py::dict input_dict) {
  std::unordered_map<int32_t, std::vector<pid_t>> map;
  if (!input_dict.empty()) {
    for (auto p : input_dict) {
      (void)map.emplace(toInt(p.first), toIntVector(py::reinterpret_borrow<py::list>(p.second)));
    }
  }
  return map;
}

std::pair<int64_t, int64_t> toIntPair(const py::tuple tuple) {
  std::pair<int64_t, int64_t> pair;
  if (!tuple.empty()) {
    pair = std::make_pair(toInt64((tuple)[0]), toInt64((tuple)[1]));
  }
  return pair;
}

std::vector<std::pair<int, int>> toPairVector(const py::list list) {
  std::vector<std::pair<int, int>> vector;
  if (list) {
    for (auto data : list) {
      auto l = data.cast<py::tuple>();
      if (l[1].is_none())
        vector.emplace_back(toInt64(l[0]), 0);
      else
        vector.emplace_back(toInt64(l[0]), toInt64(l[1]));
    }
  }
  return vector;
}

Status ToJson(const py::handle &padded_sample, nlohmann::json *const padded_sample_json,
              std::map<std::string, std::string> *sample_bytes) {
  for (const py::handle &key : padded_sample) {
    if (py::isinstance<py::bytes>(padded_sample[key])) {
      (*sample_bytes)[py::str(key).cast<std::string>()] = padded_sample[key].cast<std::string>();
      // py::str(key) enter here will loss its key name, so we create an unuse key for it in json, to pass ValidateParam
      (*padded_sample_json)[py::str(key).cast<std::string>()] = nlohmann::json::object();
    } else {
      nlohmann::json obj_json;
      if (padded_sample[key].is_none()) {
        obj_json = nullptr;
      } else if (py::isinstance<py::int_>(padded_sample[key])) {
        obj_json = padded_sample[key].cast<int64_t>();
      } else if (py::isinstance<py::float_>(padded_sample[key])) {
        obj_json = padded_sample[key].cast<double>();
      } else if (py::isinstance<py::str>(padded_sample[key])) {
        obj_json = padded_sample[key].cast<std::string>();  // also catch py::bytes
      } else {
        LOG(ERROR) << "Python object convert to json failed: " << py::cast<std::string>(padded_sample[key]);
        RETURN_STATUS_SYNTAX_ERROR("Python object convert to json failed");
      }
      (*padded_sample_json)[py::str(key).cast<std::string>()] = obj_json;
    }
  }
  return Status::OK();
}

}  // namespace minddata
