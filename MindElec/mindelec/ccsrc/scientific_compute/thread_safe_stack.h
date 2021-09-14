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

#ifndef MINDELEC_CCSRC_SCIENTIFIC_COMPUTE_THREAD_SAFE_STACK_H
#define MINDELEC_CCSRC_SCIENTIFIC_COMPUTE_THREAD_SAFE_STACK_H

#include <iostream>
#include <mutex>
#include <future>
#include <stack>
#include <memory>
#include <utility>

template<typename T>
class threadSafeStack {
 public:
  threadSafeStack() = default;
  threadSafeStack(const threadSafeStack &other);
  threadSafeStack& operator=(const threadSafeStack &&other) = delete;
  threadSafeStack(const threadSafeStack &&other)  noexcept;

  void push(T new_value);

  std::shared_ptr<T> pop();

  bool empty() const;

 private:
  std::stack<T> data;
  mutable std::mutex data_mtx;
};

#endif  // MINDELEC_CCSRC_SCIENTIFIC_COMPUTE_THREAD_SAFE_STACK_H

