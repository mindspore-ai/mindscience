"""
utils function
"""
#pylint: disable=W1203, W1202

import logging
import os
import re
import shutil

logger = logging.getLogger()


def construct_abs_path(workdir, filename, layer, resolution):
    filename = re.sub(r'\{level\}', str(layer), filename)
    filename = re.sub(r'\{resolution\}', str(resolution), filename)
    abs_path = os.path.join(workdir, filename)
    return abs_path


def get_basic_env_info(config):
    input_path = config["input_data"]
    level = config["level"]
    resolution = config["resolution"]
    output_path, tmp_path = obtain_output_tmp_relative_path(input_path)
    return input_path, output_path, tmp_path, level, resolution


def obtain_output_tmp_relative_path(input_path):
    input_path = input_path.rstrip('/')
    parent_path = os.path.abspath(os.path.join(input_path, os.path.pardir))
    output_path = os.path.join(parent_path, "output")
    tmp_path = os.path.join(parent_path, "tmp")
    return output_path, tmp_path


def make_dir(config):
    """make output and temp directory"""
    _, output_path, tmp_path, _, _ = get_basic_env_info(config)
    try:
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        if os.path.exists(tmp_path):
            shutil.rmtree(tmp_path, True)
        if not os.path.exists(tmp_path):
            os.mkdir(tmp_path)
    except OSError as e:
        logger.info("mkdir dir={} or {} failed. error={}".format(output_path, tmp_path, e))
