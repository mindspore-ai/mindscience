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

#ifndef MINDELEC_CCSRC_SCIENTIFIC_COMPUTE_POINTCLOUD_UTILS_H_
#define MINDELEC_CCSRC_SCIENTIFIC_COMPUTE_POINTCLOUD_UTILS_H_

#include <unordered_map>
#include <string>

#include "mindelec/ccsrc/include/status.h"

namespace minddata {
namespace scientific {
namespace pointcloud {

/// \brief Possible values for physical quantity in pointcloud generation with material scenario
/// At present we support those physical quantities Maxwell equation concerned about
enum class PhysicalQuantity {
  kMu = 0,        // <Magnetic permeability>
  kEpsilon = 1,   // <Permittivity>
  kSigma = 2,     // <Electrical conductivity>
  kTanD = 3       // <Dielectric loss>
};

extern std::unordered_map<PhysicalQuantity, std::string> physical_name2string_map;

// Strip a string by strip string key_S
// @param std::string key_s - key string used for strip origin string
// @param std::string *s - origin string
// @return Status - The status code returned
Status Strip(const std::string &key_s, std::string *s);

// Detect a string only start with a specific prefix
// @param std::string line - the string to be detected
// @param std::string prefix - start with prefix
// @return bool - Whether one string start with prefix
bool IsOnlyStartWith(const std::string &line, const std::string &prefix);

}  // namespace pointcloud
}  // namespace scientific
}  // namespace minddata

#endif  // MINDELEC_CCSRC_SCIENTIFIC_COMPUTE_POINTCLOUD_UTILS_H_
