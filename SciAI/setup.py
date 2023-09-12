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
# ==============================================================================
"""Setup."""

import os
import yaml
from setuptools import setup
from setuptools import find_packages


def read_version():
    """generate python file"""
    version_file = os.path.join(cur_dir, './', 'version.txt')
    with open(version_file, 'r') as f:
        version_ = f.readlines()[-1].strip()
    return version_


def get_version(package_id):
    """to get package version"""
    output = ""
    if package_id == 0:
        name = "MindFlow"
    elif package_id == 1:
        name = "MindElec"
    elif package_id == 2:
        name = "MindSPONGE"
    with open("./sciai/model/package_version.yaml", 'r') as file:
        all_info = yaml.safe_load(file)
        output = all_info.get(name).get("branch") + ":" + all_info.get(name).get("commit_id")
    print(output)


def get_model_path(model_type):
    """to get model path"""
    output = ""
    model_list = []
    if model_type == 1:
        model_list = ["MindElec"]
    elif model_type == 0:
        model_list = ["MindFlow"]
    else:
        model_list = ["MindElec", "MindFlow"]
    with open("./sciai/model/model_status.yaml", 'r') as file:
        all_info = yaml.safe_load(file)
        for model_name in all_info.keys():
            if all_info.get(model_name).get("kit") in model_list:
                output += model_name + ":" + all_info.get(model_name).get("model_path").replace("mindscience", "..")\
                          + ","
    print(output.strip(","))


def _write_version(file):
    file.write("__version__ = '{}'\n".format(version))


def build_dependencies():
    """generate python file"""
    version_file = os.path.join(pkg_dir, 'sciai', 'version.py')
    with open(version_file, 'w') as f:
        _write_version(f)

    version_file = os.path.join(cur_dir, 'sciai', 'version.py')
    with open(version_file, 'w') as f:
        _write_version(f)


if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    pkg_dir = os.path.join(cur_dir, 'build')
    package_name = os.getenv('ME_PACKAGE_NAME').replace("\n", "")
    version = read_version()

    build_dependencies()

    required_package = [
        'matplotlib >= 3.4.0',
        'mpi4py == 3.1.4',
        'numpy >= 1.17.0',
        'Ofpp == 0.1',
        'pyDOE == 0.3.8',
        'seaborn >= 0.11.1',
        'scikit-learn >= 1.0.2',
        'scikit-optimize >= 0.8.1',
        'scipy >= 1.7.0',
        'tikzplotlib == 0.10.1',
        'filelock >= 3.12.2',
        'pyyaml >= 6.0',
    ]

    package_data = {
        '': [
            '*.so*',
            '*.pyd',
            'bin/*',
            'lib/*.so*',
            'lib/*.a',
            'include/*',
            'build_info.txt',
            '**/*.json',
            '**/*.txt',
            '**/*.yaml',
            '**/*.npy',
        ],
        '_c_minddata': ['lib_c_minddata*.so']
    }

    setup(
        name=package_name,  # Required
        version=version,  # Required
        author='The MindSpore Authors',
        author_email='contact@mindspore.cn',
        url='https://www.mindspore.cn/',
        download_url='https://gitee.com/mindspore/mindscience/tags',
        project_urls={
            'Sources': 'https://gitee.com/mindspore/mindscience',
            'Issue Tracker': 'https://gitee.com/mindspore/mindscience/issues',
        },
        description=
        "A fundamental framework for simulations of engineering physics",
        license='Apache 2.0',
        packages=find_packages(),
        package_data=package_data,
        include_package_data=True,
        install_requires=required_package,
        classifiers=['License :: OSI Approved :: Apache Software License'])
