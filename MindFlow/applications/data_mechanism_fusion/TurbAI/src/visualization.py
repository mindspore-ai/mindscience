# ============================================================================
# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
visualization
"""
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def r2_score(label, predict):
    """r2_score"""
    label_mean = np.mean(label)
    ss_tot = np.sum((label - label_mean) ** 2)
    ss_res = np.sum((label - predict) ** 2)
    r2_value = 1.0 - (ss_res / ss_tot)
    return r2_value


def mean_squared_error(label, predict):
    """mean_squared_error"""
    return np.mean((label - predict) ** 2, axis=0)


def plt_loss_func(epochs, data, filename, is_train=True, prefix="result_picture"):
    """
    绘制损失函数曲线
    """
    plt.cla()
    max_value = max(data)
    min_value = min(data)
    max_index = data.index(max_value)
    min_index = data.index(min_value)

    plt.plot(list(range(epochs)), data)
    plt.plot(max_index, max_value, 'ks')
    plt.plot(min_index, min_value, 'ks')
    show_max = '(' + str(max_index) + ',' + str(max_value) + ')'
    plt.annotate(show_max, xytext=(max_index, max_value), xy=(max_index, max_value))

    show_min = f"({min_index-10}, {min_value:0.1e})"
    plt.annotate(show_min, xytext=(min_index-10, min_value), xy=(min_index-10, min_value))
    plt.xlabel("epochs")

    if is_train:
        plt.ylabel("train_loss")
    else:
        plt.ylabel("test_loss")

    plt.title("loss")

    file_name = os.path.join(prefix, filename)
    plt.savefig(file_name)


def plt_error_dis_analyze(config, data, filepath):
    """
    绘制error 和 dis的分析
    """
    plt.cla()
    label = data["post_label"].values
    predict = data["post_predict"].values
    error = np.abs(label - predict) / label
    dis = data["dis"].values
    new_data = pd.DataFrame({"error": error, "dis": dis}).sort_values(by="error")
    new_data.to_csv(filepath + '.csv')
    plt.figure(figsize=(8, 10), dpi=200)

    # error -dis 平均值直方图  0.01 < Dis < 0.2
    plt.subplot(2, 1, 1)
    hist, bin_value = get_mean_error_by_condition(config["figure_bins"],
                                                  config["plt_dis_internal"], new_data)
    plt.xticks(range(0, 9), format_sci(bin_value[1:]), rotation=30)
    plt.bar(range(0, 9), hist[1:])
    for value, prob in zip(range(0, 9), hist[1:]):
        plt.text(value, prob + 0.05, f'{prob:0.0f}', size=8, ha='center', va='bottom')
    plt.xlabel('Dis', fontsize=10)
    plt.ylabel('Error', fontsize=10)
    plt.title('Error-Dis Average Hist  (0.02 < Dis < 0.2)')

    # dis - Error 平均值直方图  0 < Error < 1
    plt.subplot(2, 1, 2)
    hist, bin_value = get_mean_dis_by_condition(config["figure_bins"],
                                                config["plt_error_internal"], new_data)
    plt.xticks(range(0, 10), format_sci(bin_value), rotation=30)
    plt.bar(range(0, 10), hist)
    for value, prob in zip(range(0, 10), hist):
        plt.text(value, prob, f'{prob:0.2f}', size=8, ha='center', va='bottom')
    plt.xlabel('Error', fontsize=10)
    plt.ylabel('Dis', fontsize=10)
    plt.title('Dis-Error Average Hist (0 < Error < 1) ')

    plt.tight_layout()
    plt.savefig(filepath)
    plt.clf()


def plt_error_distribute(config, x_value, post_predict, post_label, dis, model_predict,
                         model_label, prefix, train_or_test="train"):
    """
    绘制所有点/近壁面 绝对误差/相对误差
    随机选取500个点
    """
    near_wall = train_or_test + '_near_wall'
    all_data = train_or_test + '_all'
    shock_wave = train_or_test + 'shock_wave'
    plt.cla()
    data = pd.DataFrame({"post_label": post_label, "post_predict": post_predict,
                         "model_label": model_label, "model_predict": model_predict,
                         "dis": dis, "X": x_value})

    filepath = os.path.join(prefix, all_data + '_error.png')
    plt_hist_and_line(data, filepath)

    data_near_wall = data[data["dis"] < config["near_wall_dis"]]
    filepath = os.path.join(prefix, near_wall + '_error.png')
    plt_hist_and_line(data_near_wall, filepath)

    data_shock_wave = data[(data["dis"] < config["near_wall_dis"])
                           & (data["X"] < config["shock_wave_x_max"])
                           & (data["X"] > config["shock_wave_x_min"])]
    filepath = os.path.join(prefix, shock_wave + '_error.png')
    plt_hist_and_line(data_shock_wave, filepath)

    analyze_path = os.path.join(prefix, train_or_test + '_error_dis_analyze.png')
    plt_error_dis_analyze(config, data, analyze_path)


def plt_hist_and_line(data, filepath):
    """plt_hist_and_line"""
    plt.cla()
    plt.figure(figsize=(8, 10), dpi=200)
    data_sample = data.sample(200).sort_values(by="model_label")
    # 后处理-预测折线图
    plt.subplot(4, 1, 1)
    label_sample = data_sample["post_label"].values
    predict_sample = data_sample["post_predict"].values
    data_num = len(label_sample)

    plt.plot(range(data_num), label_sample, c='g', linestyle='-')
    plt.plot(range(data_num), predict_sample, c='r', linestyle='--')
    plt.xlabel('index', fontsize=8)
    plt.ylabel('Mut', fontsize=8)
    plt.title('Postprocess Prediction and Label')

    # 后处理-相对偏差分布
    plt.subplot(4, 1, 2)
    label = data["post_label"].values
    predict = data["post_predict"].values
    error = np.abs(label - predict) / label

    gap_data = [5 * x for x in range(0, 20)]
    hist, bin_value = np.histogram(error, gap_data)
    other_data = len(error) - sum(hist)
    hist = np.append(hist, [other_data])
    bin_value = np.append(bin_value, [101])
    hist_prob = [x / len(error) for x in hist]
    plt.bar(bin_value[1:], hist_prob, alpha=1, width=0.8)

    for value, prob in zip(bin_value[1:], hist_prob):
        plt.text(value, prob + 0.05, f'{prob:0.2f}', size=8, ha='center', va='bottom')
    plt.xlabel('Relative Error Percent', fontsize=8)
    plt.ylabel('Cell Number Percent', fontsize=8)
    plt.title('Postprocess Relative Error Distribution')

    # 模型输出-预测折线图
    plt.subplot(4, 1, 3)
    label_sample = data_sample["model_label"].values
    predict_sample = data_sample["model_predict"].values
    data_num = len(label_sample)

    plt.plot(range(data_num), label_sample, c='g', linestyle='-')
    plt.plot(range(data_num), predict_sample, c='r', linestyle='--')
    plt.xlabel('index', fontsize=8)
    plt.ylabel('Mut', fontsize=8)
    plt.title('Model Out Prediction and Label')

    # 模型输出-相对偏差分布
    plt.subplot(4, 1, 4)
    model_label = data["model_label"].values
    model_predict = data["model_predict"].values
    error = np.abs(model_label - model_predict) / model_label * 100
    gap_data = [5 * x for x in range(0, 20)]
    hist, bin_value = np.histogram(error, gap_data)

    other_data = len(error) - sum(hist)
    hist = np.append(hist, [other_data])
    bin_value = np.append(bin_value, [101])
    hist_prob = [x / len(error) for x in hist]
    plt.bar(bin_value[1:], hist_prob, alpha=1, width=0.8)
    for value, prob in zip(bin_value[1:], hist_prob):
        plt.text(value, prob + 0.05, f'{prob:0.2f}', size=8, ha='center', va='bottom')
    plt.xlabel('Relative Error Percent', fontsize=8)
    plt.ylabel('Cell Number Percent', fontsize=8)
    plt.title('Model Out Relative Error Distribution')

    # R2
    condition = re.split('/|_error', filepath)[-2]
    r2_value = r2_score(label, predict)
    print(condition + f":   R2 Score is {r2_value}")

    # MSE
    mse = mean_squared_error(model_label, model_predict)
    print(condition + f":  MSE Score is {mse}")

    plt.tight_layout()
    plt.savefig(filepath)
    plt.clf()


def get_mean_error_by_condition(bins, internal, data):
    """get_mean_error_by_condition"""
    error_list = []
    for i in range(1, bins + 1):
        data_out = data[(data['dis'] < internal * (i + 1)) & (data['dis'] >= internal * i)]
        error_list.append(data_out['error'].mean())
    return error_list, [internal * x for x in range(1, bins + 1)]


def get_mean_dis_by_condition(bins, internal, data):
    """get_mean_dis_by_condition"""
    error_list = []
    for i in range(1, bins + 1):
        data_out = data[(data['error'] < internal * (i + 1)) & (data['error'] >= internal * i)]
        error_list.append(data_out['dis'].mean())
    return error_list, [internal * x for x in range(1, bins + 1)]


def format_sci(data_list):
    """format_sci"""
    out = []
    for data in data_list:
        out.append(f"{data:0.1E}")
    return out
