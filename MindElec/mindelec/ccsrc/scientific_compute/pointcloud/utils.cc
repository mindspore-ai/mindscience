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
#include "mindelec/ccsrc/scientific_compute/pointcloud/utils.h"

namespace minddata {
namespace scientific {
namespace pointcloud {

std::unordered_map<PhysicalQuantity, std::string> physical_name2string_map {
  {PhysicalQuantity::kMu, ".Mu"}, {PhysicalQuantity::kEpsilon, ".Epsilon"},
  {PhysicalQuantity::kSigma, ".Sigma"}, {PhysicalQuantity::kTanD, ".TanD"}
};

Status Strip(const std::string &key_s, std::string *s) {
  RETURN_OK_IF_TRUE(s->empty());
  s->erase(0, s->find_first_not_of(key_s));
  s->erase(s->find_last_not_of(key_s) + 1);
  return Status::OK();
}

bool IsOnlyStartWith(const std::string &line, const std::string &prefix) {
  size_t prefix_length = prefix.size();
  if (line.rfind(prefix, 0) == 0) {
    if (line[prefix_length] == ' ') {
      return true;
    }
  }
  return false;
}

}  // namespace pointcloud
}  // namespace scientific
}  // namespace minddata
