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

#ifndef MINDELEC_CCSRC_API_PYTHON_PYBIND_CONVERSION_H_
#define MINDELEC_CCSRC_API_PYTHON_PYBIND_CONVERSION_H_

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <unordered_map>
#include <vector>
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/stl_bind.h"

#include "nlohmann/json.hpp"

#include "mindelec/ccsrc/api/python/pybind_register.h"
#include "mindelec/ccsrc/include/status.h"

namespace py = pybind11;

namespace minddata {

float toFloat(const py::handle &handle);

int toInt(const py::handle &handle);

int64_t toInt64(const py::handle &handle);

bool toBool(const py::handle &handle);

std::string toString(const py::handle &handle);

std::set<std::string> toStringSet(const py::list list);

std::map<std::string, int32_t> toStringMap(const py::dict dict);

std::vector<std::string> toStringVector(const py::list list);

std::vector<pid_t> toIntVector(const py::list input_list);

std::unordered_map<int32_t, std::vector<pid_t>> toIntMap(const py::dict input_dict);

std::pair<int64_t, int64_t> toIntPair(const py::tuple tuple);

std::vector<std::pair<int, int>> toPairVector(const py::list list);

Status ToJson(const py::handle &padded_sample, nlohmann::json *const padded_sample_json,
              std::map<std::string, std::string> *sample_bytes);

}  // namespace minddata
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_API_PYTHON_PYBIND_CONVERSION_H_
