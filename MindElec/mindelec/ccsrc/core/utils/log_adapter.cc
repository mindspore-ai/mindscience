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

namespace minddata {

static std::string GetEnv(const std::string &envvar) {
  const char *value = ::getenv(envvar.c_str());

  if (value == nullptr) {
    return std::string();
  }

  return std::string(value);
}

void common_log_init(void) {
  // do not use glog predefined log prefix
  FLAGS_log_prefix = true;
  FLAGS_logbufsecs = 0;
  // set default log level to WARNING
  if (minddata::GetEnv("GLOG_v").empty()) {
    FLAGS_v = minddata::WARNING;
  }

  // set default log file mode to 0640
  if (minddata::GetEnv("GLOG_logfile_mode").empty()) {
    FLAGS_logfile_mode = 0640;
  }
  std::string logtostderr = minddata::GetEnv("GLOG_logtostderr");

  // default print log to screen
  if (logtostderr.empty()) {
    FLAGS_logtostderr = true;
  } else if (logtostderr == "0" && minddata::GetEnv("GLOG_log_dir").empty()) {
    FLAGS_logtostderr = true;
    // LOG(WARNING) << "`GLOG_log_dir` is not set, output log to screen.";
  }
}


// shared lib init hook

void minddata_log_init(void) {
  static bool is_glog_initialzed = false;
  if (!is_glog_initialzed) {
    google::InitGoogleLogging("minddata");
    FLAGS_alsologtostderr = 1;
    is_glog_initialzed = true;
  }
  common_log_init();
}

}  // namespace minddata
