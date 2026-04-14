# Copyright 2021 Huawei Technologies Co., Ltd
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
"""train"""
from __future__ import print_function
import os, sys
import argparse
import numpy as np
import mindspore as ms
from mindspore import context, set_seed

from src.Config import get_options
from src.dataset import *
from src.utils import *

set_seed(123456)
np.random.seed(123456)


def parse_args():
    '''Parse input args'''
    parser = argparse.ArgumentParser(description="ML-DFTXC model train")
    parser.add_argument("--config", type=str, default="./config/train.cfg")
    parser.add_argument("--device_target", type=str, default="Ascend", choices=["GPU", "Ascend", "CPU"],
                        help="The target device to run, support 'Ascend', 'GPU', 'CPU")
    parser.add_argument("--device_id", type=int, default=0, help="ID of the target device")
    parser.add_argument("--mode", type=str, default="GRAPH", choices=["GRAPH", "PYNATIVE"],
                        help="Running in GRAPH_MODE OR PYNATIVE_MODE")
    input_args = parser.parse_args()
    return input_args

def main():
    args = parse_args()
    options = get_options(args.config)
    for key in options.keys():
        print(key, options[key])

    context.set_context(mode=context.GRAPH_MODE if args.mode.upper().startswith("GRAPH") else context.PYNATIVE_MODE,
                        device_target=args.device_target,
                        device_id=args.device_id)
    print(f"Running in {args.mode.upper()} mode, using device id: {args.device_id}.")

    logger = Logger(options["log_path"], to_stdout=options["verbose"])
    logger.log("========Task Start========")

    train_set_loader, validate_set_loader = \
            get_train_and_validate_set(options)

    model = get_model(options["model"])
    if "restart" in options.keys():
        load_model(model, options["restart"])

    if options["loss_function"] == "MSELoss_zsym":
        if "zsym_coef" in options.keys():
            loss_func = get_list_item(LOSS_FUNC_LIST, options["loss_function"])(float(options["zsym_coef"]))
        else:
            loss_func = get_list_item(LOSS_FUNC_LIST, options["loss_function"])()
    else:
        loss_func = get_list_item(LOSS_FUNC_LIST, options["loss_function"])()
    loss_func.size_average = True

    optimiser = get_list_item(OPTIM_LIST, options["optimiser"])(model.trainable_params(), learning_rate=options["learning_rate"])

    # start logger
    logger.log(str(model), "main")
    logger.log("Max iteration: %d" % (options["max_epoch"]), "main")
    logger.log("Learning rate: %e" % (options["learning_rate"]), "main")
    logger.log("Loss function", "main")
    logger.log(str(loss_func), "main")
    logger.log("Optimiser", "main")
    logger.log(str(optimiser), "main")
    logger.log("Model saved to %s" % (options["model_save_path"]), "main")
    os.system("mkdir -p %s" % (options["model_save_path"][:options["model_save_path"].rfind('/')]))

    n_restart = int(options["n_restart"]) if "n_restart" in options.keys() else 20

    # Train!!
    logger.log("Train start", "main")
    
    train_x = np.array(range(options["max_epoch"]))
    train_y = np.zeros(len(train_x), dtype=np.float32)
    validate_x = np.array(range(options["max_epoch"]))
    validate_y = np.zeros(len(validate_x), dtype=np.float32)

    for epoch in range(options["max_epoch"]):
        loss_on_train = train(epoch, train_set_loader, model, loss_func, optimiser, logger)
        loss_on_validate = validate(epoch, validate_set_loader, model, loss_func, logger)

        train_y[epoch] = loss_on_train
        validate_y[epoch] = loss_on_validate

        if np.argmin(validate_y[:epoch+1]) == epoch:
            save_model(model, options["model_save_path"] + f"_epoch{epoch+1}")

    save_model(model, options["model_save_path"])
    logger.log("Model saved.", "main")
    logger.log("========Task Finish========")

if __name__ == "__main__":
    print("pid:", os.getpid())
    main()

