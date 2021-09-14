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

#ifndef MINDELEC_CCSRC_STATUS_H
#define MINDELEC_CCSRC_STATUS_H

#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#if defined(__GNUC__) || defined(__clang__)
#define DEPRECATED __attribute__((deprecated))
#elif defined(_MSC_VER)
#define DEPRECATED __declspec(deprecated)
#else
#pragma message("WARNING: You need to implement DEPRECATED for this compiler")
#define DEPRECATED
#endif

namespace minddata {

enum CompCode : uint32_t {
  kMD = 0x10000000u,
};

enum StatusCode : uint32_t {
  kSuccess = 0,
  // MD
  kMDOutOfMemory = kMD | 1,
  kMDShapeMisMatch = kMD | 2,
  kMDInterrupted = kMD | 3,
  kMDNoSpace = kMD | 4,
  kMDPyFuncException = kMD | 5,
  kMDDuplicateKey = kMD | 6,
  kMDPythonInterpreterFailure = kMD | 7,
  kMDTDTPushFailure = kMD | 8,
  kMDFileNotExist = kMD | 9,
  kMDProfilingError = kMD | 10,
  kMDBoundingBoxOutOfBounds = kMD | 11,
  kMDBoundingBoxInvalidShape = kMD | 12,
  kMDSyntaxError = kMD | 13,
  kMDTimeOut = kMD | 14,
  kMDBuddySpaceFull = kMD | 15,
  kMDNetWorkError = kMD | 16,
  kMDNotImplementedYet = kMD | 17,
  // Make this error code the last one. Add new error code above it.
  kMDUnexpectedError = kMD | 127,
};

inline std::vector<char> StringToChar(const std::string &s) { return std::vector<char>(s.begin(), s.end()); }

inline std::string CharToString(const std::vector<char> &c) { return std::string(c.begin(), c.end()); }

class Status {
 public:
  Status();
  inline Status(enum StatusCode status_code, const std::string &status_msg = "");  // NOLINT(runtime/explicit)
  inline Status(const StatusCode code, int line_of_code, const char *file_name, const std::string &extra = "");

  ~Status() = default;

  enum StatusCode StatusCode() const;
  inline std::string ToString() const;

  int GetLineOfCode() const;
  inline std::string GetErrDescription() const;
  inline std::string SetErrDescription(const std::string &err_description);

  friend std::ostream &operator<<(std::ostream &os, const Status &s);

  bool operator==(const Status &other) const;
  bool operator==(enum StatusCode other_code) const;
  bool operator!=(const Status &other) const;
  bool operator!=(enum StatusCode other_code) const;

  explicit operator bool() const;
  explicit operator int() const;

  static Status OK();

  bool IsOk() const;

  bool IsError() const;

  static inline std::string CodeAsString(enum StatusCode c);

 private:
  // api without std::string
  explicit Status(enum StatusCode status_code, const std::vector<char> &status_msg);
  Status(const enum StatusCode code, int line_of_code, const char *file_name, const std::vector<char> &extra);
  std::vector<char> ToCString() const;
  std::vector<char> GetErrDescriptionChar() const;
  std::vector<char> SetErrDescription(const std::vector<char> &err_description);
  static std::vector<char> CodeAsCString(enum StatusCode c);

  struct Data;
  std::shared_ptr<Data> data_;
};

Status::Status(enum StatusCode status_code, const std::string &status_msg)
    : Status(status_code, StringToChar(status_msg)) {}
Status::Status(const enum StatusCode code, int line_of_code, const char *file_name, const std::string &extra)
    : Status(code, line_of_code, file_name, StringToChar(extra)) {}
std::string Status::ToString() const { return CharToString(ToCString()); }
std::string Status::GetErrDescription() const { return CharToString(GetErrDescriptionChar()); }
std::string Status::SetErrDescription(const std::string &err_description) {
  return CharToString(SetErrDescription(StringToChar(err_description)));
}
std::string Status::CodeAsString(enum StatusCode c) { return CharToString(CodeAsCString(c)); }


#define RETURN_IF_NOT_OK(_s) \
  do {                       \
    Status __rc = (_s);      \
    if (__rc.IsError()) {    \
      return __rc;           \
    }                        \
  } while (false)

#define RETURN_STATUS_UNEXPECTED(_e)                                       \
  do {                                                                     \
    return Status(StatusCode::kMDUnexpectedError, __LINE__, __FILE__, _e); \
  } while (false)

#define CHECK_FAIL_RETURN_UNEXPECTED(_condition, _e)                         \
  do {                                                                       \
    if (!(_condition)) {                                                     \
      return Status(StatusCode::kMDUnexpectedError, __LINE__, __FILE__, _e); \
    }                                                                        \
  } while (false)

#define CHECK_FAIL_RETURN_SYNTAX_ERROR(_condition, _e)                   \
  do {                                                                   \
    if (!(_condition)) {                                                 \
      return Status(StatusCode::kMDSyntaxError, __LINE__, __FILE__, _e); \
    }                                                                    \
  } while (false)

#define RETURN_UNEXPECTED_IF_NULL(_ptr)                                         \
  do {                                                                          \
    if ((_ptr) == nullptr) {                                                    \
      std::string err_msg = "The pointer[" + std::string(#_ptr) + "] is null."; \
      RETURN_STATUS_UNEXPECTED(err_msg);                                        \
    }                                                                           \
  } while (false)

#define RETURN_OK_IF_TRUE(_condition) \
  do {                                \
    if (_condition) {                 \
      return Status::OK();            \
    }                                 \
  } while (false)

#define RETURN_STATUS_SYNTAX_ERROR(_e)                                 \
  do {                                                                 \
    return Status(StatusCode::kMDSyntaxError, __LINE__, __FILE__, _e); \
  } while (false)

}  // namespace minddata

#endif  // MINDELEC_CCSRC_STATUS_H
