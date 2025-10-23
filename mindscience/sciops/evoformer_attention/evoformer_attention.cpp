/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "./ms_extension.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "mindspore/ccsrc/pyboost/op_runner.h"
#include "mindspore/ccsrc/debug/profiler/profiler.h"
#include "runtime/pynative/op_runner.h"
#include "runtime/pynative/op_executor.h"
#include "mindspore/ccsrc/include/common/runtime_conf/runtime_conf.h"
#include "mindspore/ccsrc/runtime/pynative/task/device_task.h"
#include "mindspore/ccsrc/plugin/res_manager/ascend/stream_manager/ascend_stream_manager.h"
#include "mindspore/ops/kernel/ascend/pyboost/aclnn_utils.h"

using BaseTensorPtr = mindspore::tensor::TensorPtr;
using BaseTensorPtrList = mindspore::tensor::TensorPtrList;

namespace mindspore::kernel::pyboost {
void CustomPyboostExecutor(const std::string &opname, const BaseTensorPtrList &inputs, const BaseTensorPtrList &outputs,
                           const std::function<void()> &exec_func) {
  mindspore::runtime::ProfilerRecorder profiler(mindspore::runtime::ProfilerModule::kPynative,
                                                mindspore::runtime::ProfilerEvent::kRunOp, opname);
  auto stream_id = PyBoostUtils::cur_stream_id();
  auto device_context = runtime::OpRunner::GetDeviceContext("Ascend");
  PyBoostUtils::PrepareOpInputs(device_context, stream_id, inputs);
  PyBoostUtils::PrepareOpOutputs(device_context, stream_id, outputs);
  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([device_context, inputs, outputs, exec_func]() {
      MS_LOG(DEBUG) << "Run device task Add start";
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, inputs);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);
      // LAUNCH_ACLNN(aclnnMul, device_context, op->stream_id(), x, y, outputs[0]);
      exec_func();
      MS_LOG(DEBUG) << "Run device task end";
    }));
}

BaseTensorPtr npu_evoformer_attention(
  const BaseTensorPtr &query, const BaseTensorPtr &key, const BaseTensorPtr &value,
  const std::optional<BaseTensorPtr> &realShiftOptional, const std::optional<BaseTensorPtr> &dropMaskOptional,
  const std::optional<BaseTensorPtr> &paddingMaskOptional, const std::optional<BaseTensorPtr> &attenMaskOptional,
  const std::optional<BaseTensorPtr> &prefixOptional, const std::optional<double> &scaleValueOptional,
  const std::optional<double> &keepProbOptional, const std::optional<int64_t> preTokensOptional,
  const std::optional<int64_t> nextTokensOptional, const std::optional<int64_t> &headNum,
  const std::optional<string> &inputLayout, const std::optional<int64_t> &innerPreciseOptional,
  const std::optional<int64_t> &sparseModeOptional) {
  auto stream_id = PyBoostUtils::cur_stream_id();
  auto device_context = runtime::OpRunner::GetDeviceContext("Ascend");
  BaseTensorPtrList inputs = {query, key, value};
  if (realShiftOptional.has_value()) {
    inputs.emplace_back(realShiftOptional.value());
  }
  if (dropMaskOptional.has_value()) {
    inputs.emplace_back(dropMaskOptional.value());
  }
  if (paddingMaskOptional.has_value()) {
    inputs.emplace_back(paddingMaskOptional.value());
  }
  if (attenMaskOptional.has_value()) {
    inputs.emplace_back(attenMaskOptional.value());
  }
  if (prefixOptional.has_value()) {
    inputs.emplace_back(prefixOptional.value());
  }

  double scaleValue = scaleValueOptional.value_or(1.0);
  double keepProb = keepProbOptional.value_or(1.0);
  int64_t preToken = preTokensOptional.value_or(2147483647);
  int64_t nextToken = nextTokensOptional.value_or(2147483647);
  int64_t head_num = headNum.value();
  string input_layout = inputLayout.value();
  int64_t inner_precise = innerPreciseOptional.value_or(0);
  int64_t sparse_mode = sparseModeOptional.value_or(0);
  BaseTensorPtr softmax_max;
  BaseTensorPtr softmax_sum;
  BaseTensorPtr softmax_out;
  BaseTensorPtr attention_out;
  ShapeVector softmax_max_sum_shape;
  ShapeVector softmax_out_shape;

  if (input_layout == "SBH") {
    softmax_max_sum_shape = {query->shape()[1], head_num, query->shape()[0], 8};
    softmax_out_shape = {query->shape()[1], head_num, query->shape()[0], query->shape()[0]};
  } else if (input_layout == "BSH" || input_layout == "BSND") {
    softmax_max_sum_shape = {query->shape()[0], head_num, query->shape()[1], 8};
    softmax_out_shape = {query->shape()[0], head_num, query->shape()[1], query->shape()[0]};
  } else if (input_layout == "BNSD") {
    softmax_max_sum_shape = {query->shape()[0], head_num, query->shape()[2], 8};
    softmax_out_shape = {query->shape()[0], head_num, query->shape()[2], query->shape()[0]};
  }

  attention_out = std::make_shared<Tensor>(query->data_type(), query->shape());
  softmax_max = std::make_shared<Tensor>(kNumberTypeFloat32, softmax_max_sum_shape);
  softmax_sum = std::make_shared<Tensor>(kNumberTypeFloat32, softmax_max_sum_shape);
  softmax_out = std::make_shared<Tensor>(query->data_type(), softmax_out_shape);
  BaseTensorPtrList result = {softmax_max, softmax_sum, softmax_out, attention_out};
  CustomPyboostExecutor("aclnnEvoformerAttention", inputs, result, [=]() {
    LAUNCH_ACLNN(aclnnEvoformerAttention, device_context, stream_id,
                 query, key, value,
                 realShiftOptional, dropMaskOptional, paddingMaskOptional,
                 attenMaskOptional, prefixOptional, scaleValue, keepProb,
                 preToken, nextToken, head_num, input_layout, inner_precise,
                 sparse_mode, softmax_max, softmax_sum, softmax_out, attention_out);
  });
  return attention_out;
}
}  // namespace mindspore::kernel::pyboost

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_evoformer_attention", &mindspore::kernel::pyboost::npu_evoformer_attention, "npu_evoformer_attention",
        pybind11::arg("query"), pybind11::arg("key"), pybind11::arg("value"),
        pybind11::arg("realShiftOptional"), pybind11::arg("dropMaskOptional"), pybind11::arg("paddingMaskOptional"),
        pybind11::arg("attenMaskOptional"), pybind11::arg("prefixOptional"), pybind11::arg("scaleValueOptional"),
        pybind11::arg("keepProbOptional"), pybind11::arg("preTokensOptional"), pybind11::arg("nextTokensOptional"),
        pybind11::arg("headNum"), pybind11::arg("inputLayout"), pybind11::arg("innerPreciseOptional"),
        pybind11::arg("sparseModeOptional"));
}
