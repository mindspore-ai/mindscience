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

#include <stdlib.h>
#include "glog/logging.h"

#ifndef MINDELEC_CCSRC_CORE_UTILS_LOG_ADAPTER_H
#define MINDELEC_CCSRC_CORE_UTILS_LOG_ADAPTER_H
#include <string>
#define LOG_HDR_FILE_REL_PATH "mindelec/ccsrc/core/utils/log_adapter.h"

// Get start index of file relative path in __FILE__
static constexpr size_t GetRelPathPos() noexcept {
  return sizeof(__FILE__) > sizeof(LOG_HDR_FILE_REL_PATH) ? sizeof(__FILE__) - sizeof(LOG_HDR_FILE_REL_PATH) : 0;
}

namespace minddata {

enum MDLogLevel : int { DEBUG = 0, INFO, WARNING, ERROR, EXCEPTION };

void minddata_log_init(void);

void common_log_init(void);

void common_log_init(void);

}  // namespace minddata

#endif  // MINDELEC_CCSRC_CORE_UTILS_LOG_ADAPTER_H
