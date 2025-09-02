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

#include <string>
#include "ms_extension/api.h"
using std::string;
using ms::pynative::FFTParam;
using ms::pynative::asdFftDirection;
using ms::pynative::asdFft1dDimType;
using ms::pynative::asdFftType;

ms::Tensor GetResultTensor(const ms::Tensor &t, const FFTParam &param) {
    auto type_id = mindspore::TypeId::kNumberTypeComplex64;
    if (param.fftType == asdFftType::ASCEND_FFT_C2R) {
        type_id = mindspore::TypeId::kNumberTypeFloat32;
    }
    // for fft and ifft, the output shape is equal to input shape
    ShapeVector out_shape(t.shape());

    // for rfft, the output shape is (batch_size, x_size, (y_size // 2) + 1)
    if (param.fftType == asdFftType::ASCEND_FFT_R2C && param.direction == asdFftDirection::ASCEND_FFT_FORWARD) {
        out_shape.back() = (out_shape.back() / 2) + 1;
    }
    // for irfft, the output shape is (batch_size, x_size, fftYSize or fftXSize1d)
    if (param.fftType == asdFftType::ASCEND_FFT_C2R && param.direction == asdFftDirection::ASCEND_FFT_BACKWARD) {
        out_shape.back() = param.fftYSize == 0 ? param.fftXSize : param.fftYSize;
    }
    return ms::Tensor(type_id, out_shape);
}

ms::Tensor exec_asdsip_fft(const string &op_name, const FFTParam &param, const ms::Tensor &input) {
    MS_LOG(INFO) << "Run device task LaunchAsdSipFFT start, op_name: " << op_name << ", param.fftX: " << param.fftXSize
        << ", param.fftY: " << param.fftYSize << ", param.fftType: " << param.fftType << ", param.direction: "
        << param.direction << ", param.batchSize: " << param.batchSize << ", param.dimType: " << param.dimType;
    auto output = GetResultTensor(input, param);
    ms::pynative::RunAsdSipFFTOp(op_name, param, input, output);
    MS_LOG(INFO) << "Run device task LaunchAsdSipFFT end";

    return output;
}

auto pyboost_npu_set_cache_size(int64_t cache_size) {
    MS_LOG(INFO) << "Set cache size for NPU FFT, cache_size: " << cache_size;
    ms::pynative::AsdSipFFTOpRunner::SetCacheSize(cache_size);
    MS_LOG(INFO) << "Set cache size for NPU FFT end";
}

// 1D FFT forward
auto pyboost_npu_fft_1d(const ms::Tensor &input, int64_t batch_size, int64_t x_size) {
    FFTParam param;
    param.fftXSize = x_size;
    param.fftYSize = 0;
    param.fftType = asdFftType::ASCEND_FFT_C2C;
    param.direction = asdFftDirection::ASCEND_FFT_FORWARD;
    param.batchSize = batch_size;
    param.dimType = asdFft1dDimType::ASCEND_FFT_HORIZONTAL;
    const string op_name = "asdFftExecC2C";
    return ms::pynative::PyboostRunner::Call<1>(exec_asdsip_fft, op_name, param, input);
}

// 1D FFT inverse
auto pyboost_npu_ifft_1d(const ms::Tensor &input, int64_t batch_size, int64_t x_size) {
    ms::pynative::FFTParam param;
    param.batchSize = batch_size;
    param.fftXSize = x_size;
    param.fftYSize = 0;
    param.fftType = asdFftType::ASCEND_FFT_C2C;
    param.direction = asdFftDirection::ASCEND_FFT_BACKWARD;
    param.dimType = asdFft1dDimType::ASCEND_FFT_HORIZONTAL;
    const string op_name = "asdFftExecC2C";
    return ms::pynative::PyboostRunner::Call<1>(exec_asdsip_fft, op_name, param, input);
}

// 1D RFFT forward
auto pyboost_npu_rfft_1d(const ms::Tensor &input, int64_t batch_size, int64_t x_size) {
    ms::pynative::FFTParam param;
    param.batchSize = batch_size;
    param.fftXSize = x_size;
    param.fftYSize = 0;
    param.fftType = asdFftType::ASCEND_FFT_R2C;
    param.direction = asdFftDirection::ASCEND_FFT_FORWARD;
    param.dimType = asdFft1dDimType::ASCEND_FFT_HORIZONTAL;
    const string op_name = "asdFftExecR2C";
    return ms::pynative::PyboostRunner::Call<1>(exec_asdsip_fft, op_name, param, input);
}

// 1D IRFFT inverse
auto pyboost_npu_irfft_1d(const ms::Tensor &input, int64_t batch_size, int64_t x_size) {
    ms::pynative::FFTParam param;
    param.batchSize = batch_size;
    param.fftXSize = x_size;
    param.fftYSize = 0;
    param.fftType = asdFftType::ASCEND_FFT_C2R;
    param.direction = asdFftDirection::ASCEND_FFT_BACKWARD;
    param.dimType = asdFft1dDimType::ASCEND_FFT_HORIZONTAL;
    const string op_name = "asdFftExecC2R";
    return ms::pynative::PyboostRunner::Call<1>(exec_asdsip_fft, op_name, param, input);
}

// 2D FFT forward
auto pyboost_npu_fft_2d(const ms::Tensor &input, int64_t batch_size, int64_t x_size, int64_t y_size) {
    ms::pynative::FFTParam param;
    param.fftXSize = x_size;
    param.fftYSize = y_size;
    param.fftType = asdFftType::ASCEND_FFT_C2C;
    param.direction = asdFftDirection::ASCEND_FFT_FORWARD;
    param.batchSize = batch_size;
    const string op_name = "asdFftExecC2C";
    return ms::pynative::PyboostRunner::Call<1>(exec_asdsip_fft, op_name, param, input);
}

// 2D IFFT inverse
auto pyboost_npu_ifft_2d(const ms::Tensor &input, int64_t batch_size, int64_t x_size, int64_t y_size) {
    ms::pynative::FFTParam param;
    param.batchSize = batch_size;
    param.fftXSize = x_size;
    param.fftYSize = y_size;
    param.fftType = asdFftType::ASCEND_FFT_C2C;
    param.direction = asdFftDirection::ASCEND_FFT_BACKWARD;
    const string op_name = "asdFftExecC2C";
    return ms::pynative::PyboostRunner::Call<1>(exec_asdsip_fft, op_name, param, input);
}

// 2D RFFT forward
auto pyboost_npu_rfft_2d(const ms::Tensor &input, int64_t batch_size, int64_t x_size, int64_t y_size) {
    ms::pynative::FFTParam param;
    param.batchSize = batch_size;
    param.fftXSize = x_size;
    param.fftYSize = y_size;
    param.fftType = asdFftType::ASCEND_FFT_R2C;
    param.direction = asdFftDirection::ASCEND_FFT_FORWARD;
    const string op_name = "asdFftExecR2C";
    return ms::pynative::PyboostRunner::Call<1>(exec_asdsip_fft, op_name, param, input);
}

// 2D IRFFT inverse
auto pyboost_npu_irfft_2d(const ms::Tensor &input, int64_t batch_size, int64_t x_size, int64_t y_size) {
    ms::pynative::FFTParam param;
    param.batchSize = batch_size;
    param.fftXSize = x_size;
    param.fftYSize = y_size;
    param.fftType = asdFftType::ASCEND_FFT_C2R;
    param.direction = asdFftDirection::ASCEND_FFT_BACKWARD;
    const string op_name = "asdFftExecC2R";
    return ms::pynative::PyboostRunner::Call<1>(exec_asdsip_fft, op_name, param, input);
}

// Python binding module
PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
    m.doc() = "NPU FFT operations for MindSpore";

    m.def("asd_set_cache_size", &pyboost_npu_set_cache_size, "Set cache size for NPU FFT",
          pybind11::arg("cache_size"));

    // 1D FFT
    m.def("asd_fft_1d", &pyboost_npu_fft_1d, "1D FFT on NPU",
          pybind11::arg("input"),
          pybind11::arg("batch_size"),
          pybind11::arg("x_size"));

    // 1D IFFT
    m.def("asd_ifft_1d", &pyboost_npu_ifft_1d, "1D IFFT on NPU",
          pybind11::arg("input"),
          pybind11::arg("batch_size"),
          pybind11::arg("x_size"));

    // 1D RFFT
    m.def("asd_rfft_1d", &pyboost_npu_rfft_1d, "1D RFFT on NPU",
          pybind11::arg("input"),
          pybind11::arg("batch_size"),
          pybind11::arg("x_size"));

    // 1D IRFFT
    m.def("asd_irfft_1d", &pyboost_npu_irfft_1d, "1D IRFFT on NPU",
          pybind11::arg("input"),
          pybind11::arg("batch_size"),
          pybind11::arg("x_size"));

    // 2D FFT
    m.def("asd_fft_2d", &pyboost_npu_fft_2d, "2D FFT on NPU",
          pybind11::arg("input"),
          pybind11::arg("batch_size"),
          pybind11::arg("x_size"),
          pybind11::arg("y_size"));

    // 2D IFFT
    m.def("asd_ifft_2d", &pyboost_npu_ifft_2d, "2D IFFT on NPU",
          pybind11::arg("input"),
          pybind11::arg("batch_size"),
          pybind11::arg("x_size"),
          pybind11::arg("y_size"));

    // 2D RFFT
    m.def("asd_rfft_2d", &pyboost_npu_rfft_2d, "2D RFFT on NPU",
          pybind11::arg("input"),
          pybind11::arg("batch_size"),
          pybind11::arg("x_size"),
          pybind11::arg("y_size"));

    // 2D IRFFT
    m.def("asd_irfft_2d", &pyboost_npu_irfft_2d, "2D IRFFT on NPU",
          pybind11::arg("input"),
          pybind11::arg("batch_size"),
          pybind11::arg("x_size"),
          pybind11::arg("y_size"));
}
