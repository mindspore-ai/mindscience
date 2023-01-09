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
DEFINE_string(dataset, ".", "dataset name");
DEFINE_int32(device_id, 0, "device id");

void InitInputs(size_t i, const std::vector<MSTensor> &model_inputs, std::vector<MSTensor> *inputs) {
  std::string dataset_dir = FLAGS_input_path + "/" + FLAGS_dataset;
  auto f_atoms_files = GetAllFiles(dataset_dir+"/01_f_atoms");
  auto f_bonds_files = GetAllFiles(dataset_dir+"/02_f_bonds");
  auto a2b_files = GetAllFiles(dataset_dir+"/03_a2b");
  auto b2a_files = GetAllFiles(dataset_dir+"/04_b2a");
  auto b2revb_files = GetAllFiles(dataset_dir+"/05_b2revb");
  auto a2a_files = GetAllFiles(dataset_dir+"/06_a2a");
  auto a_scope_files = GetAllFiles(dataset_dir+"/07_a_scope");
  auto b_scope_files = GetAllFiles(dataset_dir+"/08_b_scope");
  auto features_batch_files = GetAllFiles(dataset_dir+"/00_features_batch");

  auto f_atoms = ReadFileToTensor(f_atoms_files[i]);
  auto f_bonds = ReadFileToTensor(f_bonds_files[i]);
  auto a2b = ReadFileToTensor(a2b_files[i]);
  auto b2a = ReadFileToTensor(b2a_files[i]);
  auto b2revb = ReadFileToTensor(b2revb_files[i]);
  auto a2a = ReadFileToTensor(a2a_files[i]);
  auto a_scope = ReadFileToTensor(a_scope_files[i]);
  auto b_scope = ReadFileToTensor(b_scope_files[i]);
  auto features_batch = ReadFileToTensor(features_batch_files[i]);

  inputs->emplace_back(model_inputs[0].Name(), model_inputs[0].DataType(), model_inputs[0].Shape(),
                       f_atoms.Data().get(), f_atoms.DataSize());
  inputs->emplace_back(model_inputs[1].Name(), model_inputs[1].DataType(), model_inputs[1].Shape(),
                       f_bonds.Data().get(), f_bonds.DataSize());
  inputs->emplace_back(model_inputs[2].Name(), model_inputs[2].DataType(), model_inputs[2].Shape(),
                       a2b.Data().get(), a2b.DataSize());
  inputs->emplace_back(model_inputs[3].Name(), model_inputs[3].DataType(), model_inputs[3].Shape(),
                       b2a.Data().get(), b2a.DataSize());
  inputs->emplace_back(model_inputs[4].Name(), model_inputs[4].DataType(), model_inputs[4].Shape(),
                       b2revb.Data().get(), b2revb.DataSize());
  inputs->emplace_back(model_inputs[5].Name(), model_inputs[5].DataType(), model_inputs[5].Shape(),
                       a2a.Data().get(), a2a.DataSize());
  inputs->emplace_back(model_inputs[6].Name(), model_inputs[6].DataType(), model_inputs[6].Shape(),
                       a_scope.Data().get(), a_scope.DataSize());
  inputs->emplace_back(model_inputs[7].Name(), model_inputs[7].DataType(), model_inputs[7].Shape(),
                       b_scope.Data().get(), b_scope.DataSize());
  inputs->emplace_back(model_inputs[8].Name(), model_inputs[8].DataType(), model_inputs[8].Shape(),
                       features_batch.Data().get(), features_batch.DataSize());
  return;
}

int main(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    if (RealPath(FLAGS_mindir_path).empty()) {
        std::cout << "Invalid mindir" << std::endl;
        return 1;
    }
    /* set context */
    auto context = std::make_shared<Context>();
    auto ascend310 = std::make_shared<mindspore::Ascend310DeviceInfo>();
    ascend310->SetDeviceID(FLAGS_device_id);
    ascend310->SetPrecisionMode("allow_fp32_to_fp16");
    context->MutableDeviceInfo().push_back(ascend310);

    /* Load graph and build model. */
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
    /* set files */
    std::string dataset_dir = FLAGS_input_path + "/" + FLAGS_dataset;
    auto preds_files = GetAllFiles(dataset_dir+"/10_preds");

    std::map<double, double> costTime_map;
    size_t size = preds_files.size();
    std::cout <<"SIZE:" << size << std::endl;
    for (size_t i = 0; i < size; ++i) {
        struct timeval start = {0};
        struct timeval end = {0};
        double startTimeMs;
        double endTimeMs;

        std::vector<MSTensor> inputs;
        std::vector<MSTensor> outputs;
        std::cout << "Start predict input files" << std::endl;
        InitInputs(i, model_inputs, &inputs, dataset_dir);
        gettimeofday(&start, nullptr);
        ret = model.Predict(inputs, &outputs);  // infer
        gettimeofday(&end, nullptr);
        if (ret != kSuccess) {
            std::cout << "Predict failed." << std::endl;
            return 1;
        }
        startTimeMs = (1.0 * start.tv_sec * CONVERT_TO_SEC + start.tv_usec) / SEC_TO_MS;
        endTimeMs = (1.0 * end.tv_sec * CONVERT_TO_SEC + end.tv_usec) / SEC_TO_MS;
        costTime_map.insert(std::pair<double, double>(startTimeMs, endTimeMs));
        int ret_ = WriteResult(preds_files[i], outputs);  // write the infer results
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
