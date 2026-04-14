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
# ==============================================================================
"""the callback functions of SKNO"""
import time

from mindearth.module import WeatherForecast



def divide_function(numerator, denominator):
    r"""
    Check denominator, and compute the divide
    """
    if isinstance(denominator, (int, float)):
        if denominator != 0:
            result = numerator / denominator
        else:
            raise ValueError("The numerator is divided by Zero!")
    else:
        if denominator.all() != 0:
            result = numerator / denominator
        else:
            raise ValueError("The numerator is divided by Zero!")
    return result



class InferenceModule(WeatherForecast):
    """
    Perform multiple rounds of model inference.
    """

    def __init__(self, model, config, logger):
        super(InferenceModule, self).__init__(model, config, logger)

    def forecast(self, inputs):
        pred_lst = []
        for _ in range(self.t_out):
            start_time = time.time()
            pred, _ = self.model(inputs)
            pred_lst.append(pred.transpose(0, 2, 3, 1).reshape(self.batch_size, self.h_size * self.w_size,
                                                               self.feature_dims).asnumpy())
            end_time = time.time()
            inputs = pred
        return pred_lst





