# Copyright 2022 Huawei Technologies Co., Ltd
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
"""utils"""
import time
import logging
import numpy as np
import mindspore as ms
import mindspore.nn as nn

from src.model import *
from src.loss import *

MODEL_LIST = {
    "cnn_lda_0": CNN_LDA_0, 
    "cnn_gga_0": CNN_GGA_0,
    "cnn_gga_1": CNN_GGA_1,
    "cnn_gga_1_zsym": CNN_GGA_1_zsym, 
    # "cnn_gga_2": CNN_GGA_2,
    # "cnn_gga_2_zsym": CNN_GGA_2_zsym, 
    # "cnn_gga_3": CNN_GGA_3,
    # "cnn_gga_3_zsym": CNN_GGA_3_zsym, 
    # "cnn_gga_5": CNN_GGA_5, 
    # "cnn_gga_5_zsym": CNN_GGA_5_zsym, 
    # "cnn_gga_7": CNN_GGA_7, 
    # "cnn_gga_7_zsym": CNN_GGA_7_zsym, 

    # "cnn_gga_1_sigmoid": CNN_GGA_1_sigmoid, 
    # "cnn_gga_2_tanh": CNN_GGA_2_tanh,
    # "cnn_gga_15_rddm_elu": CNN_GGA_15_rddm_elu, 
    # "dnn": DNN,
    # "cnn_lda_pyramid_9_30": CNN_LDA_pyramid_9_30, 
    # "cnn_rdg_1": CNN_RDG_1,
    # "cnn_rdg_1_zsym": CNN_RDG_1_zsym,
}

OPTIM_LIST = {
    "sgd": nn.SGD, 
    "dafault": nn.SGD, 
}

LOSS_FUNC_LIST = {
    "mseloss": nn.MSELoss,
    "rmseloss": RMSELoss,
    "wmseloss": WMSELoss, 
    "default": nn.MSELoss,
    "mseloss_zsym": MSELoss_zsym, 
}

class ExtendModel(nn.Cell):
    def __init__(self, model, extend_size, *args):
        super(ExtendModel, self).__init__()
        self.model = get_model(model, *args)
        # _add_extended_fc_layer(self, extend_size)

    def load_model(self, load_path):
        param_dict = ms.load_checkpoint(load_path)
        ms.load_param_into_net(self.model, param_dict)

    def construct(self, x):
        x = self.model.construct(x)
        # x = self.fc_extend(x)
        return x

def _add_extended_fc_layer(model, extend_size):
    model.fc_extend = nn.Dense(1, extend_size)
    param_iter = iter(model.fc_extend.parameters())
    # param = param_iter.next()
    param = next(param_iter)
    param.data[:] = 0.
    param.data[extend_size // 2] = 1.
    param = param_iter.next()
    param.data[:] = 0.

def get_model(model_name, *args):
    model = None
    try:
        model = MODEL_LIST[model_name.lower()](*args)
    except KeyError:
        print("No model named %s" % (model_name.lower()))
        exit(1)
    else:
        pass
    return model

def get_list_item(LIST, key):
    try:
        result = LIST[key.lower()]
    except KeyError:
        result = LIST["default"]
    else:
        pass
    return result

def train(epoch, train_set_loader, model, loss_func, optimiser, logger=None, panel=None, cuda=False):
    model.set_train()
    time_start = time.time()
    batch_size = train_set_loader.batch_size
    running_loss = 0.

    def net_forward(inputs, targets):
        logits = ms.mutable(model(inputs))
        loss = loss_func(logits, targets)
        return loss
    
    net_backward = ms.value_and_grad(net_forward, None, optimiser.parameters)

    def train_step(inputs, targets):
        loss, grads = net_backward(inputs, targets)
        optimiser(grads)
        return loss
    
    for batch_idx, data in enumerate(train_set_loader.create_dict_iterator()):
        inputs, targets = data["data"]["rho"], data["data"]["v"]
        loss = train_step(inputs, targets)
        running_loss += loss.asnumpy()

        if logger is not None and batch_idx % 50 == 49:
            logger.log("train batch %5d: [%5d/%5d]\tloss: %.8e" % 
                    (batch_idx + 1, batch_idx * batch_size, len(train_set_loader), loss.item()), "train")

    if logger is not None:
        logger.log("Epoch %5d: average loss on train: %.8e" % 
                (epoch, running_loss * batch_size / float(len(train_set_loader))), "train")
        logger.log("elapse time: %lf" % (time.time() - time_start), "train")

    return running_loss

def validate(epoch, validate_set_loader, model, loss_func, logger=None, cuda=False):
    model.set_train(False)
    time_start = time.time()
    batch_size = validate_set_loader.batch_size
    running_loss = 0.

    def net_forward(inputs, targets):
        logits = ms.mutable(model(inputs))
        loss = loss_func(logits, targets)
        return loss

    for batch_idx, data in enumerate(validate_set_loader.create_dict_iterator()):
        inputs, targets = data["data"]["rho"], data["data"]["v"]
        loss = net_forward(inputs, targets)
        running_loss += loss.asnumpy()

        if logger is not None and batch_idx % 50 == 49:
            logger.log("validate batch %5d: [%5d/%5d]\tloss: %.8e" % 
                    (batch_idx + 1, batch_idx * batch_size, len(validate_set_loader), loss.item()), "validate")

    if logger is not None:
        logger.log("Epoch %5d: average loss on validate: %.8e" % 
                (epoch, running_loss * batch_size / float(len(validate_set_loader))), "validate")
        logger.log("elapse time: %lf" % (time.time() - time_start), "validate")

    return running_loss

def test(test_set_loader, model, loss_func, logger=None, cuda=False):
    time_start = time.time()
    batch_size = test_set_loader.batch_size # should be 1
    running_loss = np.zeros([len(test_set_loader)])
    running_output = np.empty(len(test_set_loader))
    running_target = np.empty(len(test_set_loader))
    for batch_idx, data in enumerate(test_set_loader.create_dict_iterator()):
        inputs, targets = data["data"]["rho"], data["data"]["v"]

        outputs = model(inputs)
        loss = loss_func(outputs, targets)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        running_loss[batch_idx] = loss.asnumpy()
        running_output[batch_idx] = outputs.data[0][0]
        running_target[batch_idx] = targets.data[0][0]

        if logger is not None:
            logger.log("test batch %5d: [%5d/%5d]\tloss: %.8e" % 
                    (batch_idx + 1, batch_idx * batch_size, len(test_set_loader), loss.item()), "test")

    if logger is not None:
        logger.log("Average loss on test: %.8e" % 
                (np.mean(running_loss)), "test")
        logger.log("elapse time: %lf" % (time.time() - time_start), "test")

    return running_loss, running_output, running_target

# IO
def save_model(model, save_path):
    ms.save_checkpoint(model, save_path)

def load_model(model, load_path):
    param_dict = ms.load_checkpoint(load_path)
    ms.load_param_into_net(model, param_dict)

def save_ndarray(target, save_path):
    np.save(save_path, target)

# LOGGING
class Logger:
    def __init__(self, log_path, to_stdout=False):
        logging.basicConfig(filename=log_path, level=logging.INFO)

        if to_stdout:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            logging.getLogger().addHandler(ch)

        self.logger = logging.getLogger()


    def log(self, content, name=None):
        if name is None:
            self.logger.info("%s" % (content))
        else:
            self.logger.info("%s: %s" % (name, content))