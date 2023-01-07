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

#include "include/api/model.h"
#include "include/api/context.h"
#include "include/api/types.h"
#include "include/api/serialization.h"
#include "include/dataset/vision_ascend.h"
#include "include/dataset/execute.h"
#include "include/dataset/transforms.h"
#include "include/dataset/vision.h"
#include "inc/utils.h"

namespace ms = mindspore;

int main(int argc, char * argv[]) {
    std::map<double, double> costTime_map;
    // inference
    auto context = make_shared<ms::Context>();
    auto ascend310_info = make_shared<ms::Ascend310DeviceInfo>();
    ascend310_info->SetDeviceID(0);
    context->MutableDeviceInfo().push_back(ascend310_info);
    std::cout << "ascend setting done!\n";

    const auto mindir_file = "model/ufold_oneshape.mindir";
    ms::Graph graph;
    // ms::Status ret_ = ms::Serialization::Load(mindir_file, ms::ModelType::kMindIR, &graph);
    ms::Serialization::Load(mindir_file, ms::ModelType::kMindIR, &graph);
    std::cout << "load model done!\n";

    ms::Model model;
    ret = model.Build(ms::GraphCell(graph), context);
    std::cout << "build model done!\n";

    auto all_files = GetAllFiles("preprocess_Result");
    std::cout << typeid(all_files).name() << std::endl;
    if (all_files.empty()) {
        std::cout << "ERROR: no input data." << std::endl;
        return 1;
    }

    vector<ms::MSTensor> model_inputs = model.GetInputs();
    cout << "get input done!\n";

    size_t size = all_files.size();
    for (size_t i = 0; i < size; ++i) {
        struct timeval start = {0};
        struct timeval end = {0};
        double startTimeMs = 0;
        double endTimeMs = 0;
        std::vector<ms::MSTensor> inputs;
        std::vector<ms::MSTensor> outputs;
        std::cout << "==> data: " << all_files[i] << std::endl;
        ms::MSTensor data = ReadFileToTensor(all_files[i]);

        inputs.emplace_back(model_inputs[0].Name(), model_inputs[0].DataType(), model_inputs[0].Shape(),
                            data.Data().get(), data.DataSize());

        gettimeofday(&start, nullptr);

        ret = model.Predict(inputs, &outputs);
        gettimeofday(&end, nullptr);

        startTimeMs = (1.0 * start.tv_sec * 1000000 + start.tv_usec) / 1000;
        endTimeMs = (1.0 * end.tv_sec * 1000000 + end.tv_usec) / 1000;
        costTime_map.insert(std::pair<double, double>(startTimeMs, endTimeMs));
        WriteResult(all_files[i], outputs);
    }
    double average = 0.0;
    int inferCount = 0;
    for (auto iter = costTime_map.begin(); iter != costTime_map.end(); iter++) {
        average += iter->second - iter->first;
        inferCount++;
    }
    average = average / inferCount;
    std::stringstream timeCost;
    timeCost << "NN inference cost average time: " << average << " ms of infer_count " << inferCount << std::endl;
    std::cout << "NN inference cost average time: " << average << "ms of infer_count " << inferCount << std::endl;
    std::string fileName = "./time_Result" + std::string("/test_perform_static.txt");
    std::ofstream fileStream(fileName.c_str(), std::ios::trunc);
    fileStream << timeCost.str();
    fileStream.close();
    costTime_map.clear();
    return 0;
}
