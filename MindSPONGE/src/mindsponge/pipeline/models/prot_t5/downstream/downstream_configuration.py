# Copyright 2024 Huawei Technologies Co., Ltd
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
"prott5 downstream task configure"
prott5downtask_configuration = {
    "protT5_base":
        "https://gitee.com/mindspore/mindscience/raw/master/MindSPONGE/applications/model_configs/ProtT5/t5_xl.yaml",
    "protT5downtask_predict":
        "https://gitee.com/mindspore/mindscience/raw/master/MindSPONGE/applications/model_configs/ProtT5/t5_downstream_task_eval.yaml",
    "protT5downtask_train":
        "https://gitee.com/mindspore/mindscience/raw/master/MindSPONGE/applications/model_configs/ProtT5/t5_downstream_task_train.yaml"
}
