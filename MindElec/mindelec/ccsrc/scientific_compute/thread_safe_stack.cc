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

#include "mindelec/ccsrc/scientific_compute/thread_safe_stack.h"

template<typename T>
threadSafeStack<T>::threadSafeStack(const threadSafeStack &other) {
  std::lock_guard<std::mutex> lock(other.data_mtx);
  data = other.data;
}

template<typename T>
threadSafeStack<T>::threadSafeStack(const threadSafeStack &&other) noexcept {
  std::lock_guard<std::mutex> lock(other.data_mtx);
  data = std::move(other.data);
}

template<typename T>
void threadSafeStack<T>::push(T new_value) {
  std::lock_guard<std::mutex> lock(data_mtx);
  data.push(new_value);
}

template<typename T>
std::shared_ptr<T> threadSafeStack<T>::pop() {
  std::lock_guard<std::mutex> lock(data_mtx);
  if (data.empty()) {
    throw;
  }
  std::shared_ptr<T> res = std::make_shared<T>(data.top());
  data.pop();
  return res;
}

template<typename T>
bool threadSafeStack<T>::empty() const {
  std::lock_guard<std::mutex> lock(data_mtx);
  return data.empty();
}
