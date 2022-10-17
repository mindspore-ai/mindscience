/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include <sys/time.h>
#include <gflags/gflags.h>
#include <dirent.h>
#include <iostream>
#include <string>
#include <algorithm>
#include <iosfwd>
#include <vector>
#include <fstream>
#include <sstream>

#include "include/api/context.h"
#include "include/api/model.h"
#include "include/api/types.h"
#include "include/api/serialization.h"
#include "include/dataset/vision.h"
#include "include/dataset/execute.h"
#include "../inc/utils.h"

using mindspore::Context;
using mindspore::Serialization;
using mindspore::Model;
using mindspore::Status;
using mindspore::ModelType;
using mindspore::GraphCell;
using mindspore::kSuccess;
using mindspore::MSTensor;
using mindspore::dataset::Execute;

constexpr int CONVERT_TO_SEC = 1000000;
constexpr int SEC_TO_MS = 1000;

DEFINE_string(mindir_path, "", "mindir path");
DEFINE_string(input_path, ".", "dataset path");
DEFINE_int32(device_id, 0, "device id");


int main(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    if (RealPath(FLAGS_mindir_path).empty()) {
        std::cout << "Invalid mindir" << std::endl;
        return 1;
    }

    auto context = std::make_shared<Context>();
    auto ascend310 = std::make_shared<mindspore::Ascend310DeviceInfo>();
    ascend310->SetDeviceID(FLAGS_device_id);
    // ascend310->SetPrecisionMode("allow_fp32_to_fp16");
    context->MutableDeviceInfo().push_back(ascend310);
    mindspore::Graph graph;
    std::cout <<"Start load mindir" << std::endl;
    Serialization::Load(FLAGS_mindir_path, ModelType::kMindIR, &graph);

    Model model;
    std::cout << "Start build graph" << std::endl;
    Status ret = model.Build(GraphCell(graph), context);
    if (ret != kSuccess) {
        std::cout << "ERROR: Build failed." << std::endl;
        return 1;
    }
    std::cout << "Start get inputs" << std::endl;
    std::vector<MSTensor> model_inputs = model.GetInputs();
    if (model_inputs.empty()) {
        std::cout << "Invalid model, inputs is empty." << std::endl;
        return 1;
    }
    // xxxxxxxxxxxxxxxxxxxxxxxxxx
    auto all_files = GetAllFiles(FLAGS_input_path);
    if (all_files.empty()) {
        std::cout << "ERROR: no input data." << std::endl;
        return 1;
    }

    std::map<double, double> costTime_map;
    size_t size = all_files.size();
    std::cout <<"SIZE:" << size << std::endl;
    for (size_t i = 0; i < size; ++i) {
        struct timeval start = {0};
        struct timeval end = {0};
        double startTimeMs;
        double endTimeMs;
        std::vector<MSTensor> inputs;
        std::vector<MSTensor> outputs;
        std::cout << "Start predict input files:" << all_files[i] << std::endl;

        auto input = ReadFileToTensor(all_files[i]);

        inputs.emplace_back(model_inputs[0].Name(), model_inputs[0].DataType(), model_inputs[0].Shape(),
                            input.Data().get(), input.DataSize());

        gettimeofday(&start, nullptr);
        ret = model.Predict(inputs, &outputs);
        gettimeofday(&end, nullptr);
        if (ret != kSuccess) {
            std::cout << "Predict " << all_files[i] << " failed." << std::endl;
            return 1;
        }
        startTimeMs = (1.0 * start.tv_sec * CONVERT_TO_SEC + start.tv_usec) / SEC_TO_MS;
        endTimeMs = (1.0 * end.tv_sec * CONVERT_TO_SEC + end.tv_usec) / SEC_TO_MS;
        costTime_map.insert(std::pair<double, double>(startTimeMs, endTimeMs));
        int ret_ = WriteResult(all_files[i], outputs);
        if (ret_ != kSuccess) {
            std::cout << "write result failed." << std::endl;
            return 1;
        }
    }
    double average = 0.0;
    int inferCount = 0;

    for (auto iter = costTime_map.begin(); iter != costTime_map.end(); iter++) {
    double diff = 0.0;
    diff = iter->second - iter->first;
    average += diff;
    inferCount++;
    }
    average = average / inferCount;
    std::stringstream timeCost;
    timeCost << "NN inference cost average time: "<< average << " ms of infer_count " << inferCount << std::endl;
    std::cout << "NN inference cost average time: "<< average << "ms of infer_count " << inferCount << std::endl;
    std::string fileName = "./time_Result" + std::string("/test_perform_static.txt");
    std::ofstream fileStream(fileName.c_str(), std::ios::trunc);
    fileStream << timeCost.str();
    fileStream.close();
    costTime_map.clear();
    return 0;
}
