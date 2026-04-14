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
"""load config"""
try:
    import ConfigParser
except:
    import configparser as ConfigParser

import datetime 

def load_default_options():
    options = dict()
    # logger
    options["log_path"] = "AUTO"
    options["verbose"] = "True"
    options["stat_panel"] = "False" 
    
    # data 
    options["data_path"] = "data.npy"
    options["shuffle"] = "True"
    options["num_workers"] = 1
    options["input_channel"] = 4

    # NN parameter
    options["batch_size"] = 200
    options["max_epoch"] = 1000
    options["learning_rate"] = 1e-4
    options["loss_function"] = "MSELoss"
    options["optimiser"] = "SGD"
    
    # model
    options["model_save_path"] = "saved_model"

    return options

def get_options(config_file):
    config = ConfigParser.ConfigParser()
    config.read(config_file)
    options = load_default_options()
    section = config.sections()[0]
    for o in config.options(section):
        options[o] = config.get(section, o)

    parse(options)
    return options

def parse(options):
    if options["log_path"] == "AUTO":
        options["log_path"] = "log/%s_%s_%s.log" % (options["model"], str(datetime.date.today()), str(datetime.datetime.now().time()))

    if options["verbose"].lower() == "true":
        options["verbose"] = True
    else:
        options["verbose"] = False

    if options["stat_panel"].lower() == "true":
        options["stat_panel"] = True
    else:
        options["stat_panel"] = False


    # data
    if "train_set_size" in options.keys():
        options["train_set_size"] = int(options["train_set_size"])
    if "validate_set_size" in options.keys():
        options["validate_set_size"] = int(options["validate_set_size"])
    if "test_set_size" in options.keys():
        options["test_set_size"] = int(options["test_set_size"])
    if options["shuffle"].lower() == "true":
        options["shuffle"] = True
    else:
        options["shuffle"] = False
    options["num_workers"] = int(options["num_workers"])
    options["input_channel"] = int(options["input_channel"])

    # NN parameter
    options["batch_size"] = int(options["batch_size"])
    options["max_epoch"] = int(options["max_epoch"])
    options["learning_rate"] = float(options["learning_rate"])
    
    # model
    # if not options["model_save_path"].endswith(".dat"):
    #     options["model_save_path"] = "%s/%s_%s_%s.dat" % (options["model_save_path"], options["model"], str(datetime.date.today()), str(datetime.datetime.now().time()))

