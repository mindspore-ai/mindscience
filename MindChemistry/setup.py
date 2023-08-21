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
"""Setup."""

import os
from setuptools import setup
from setuptools import find_packages

cur_dir = os.path.dirname(os.path.realpath(__file__))
pkg_dir = os.path.join(cur_dir, 'build')
package_name = os.getenv('ME_PACKAGE_NAME').replace("\n", "")


def read_version():
    """generate python file"""
    version_file = os.path.join(cur_dir, './', 'version.txt')
    with open(version_file, 'r') as f:
        version_ = f.readlines()[-1].strip()
    return version_


version = read_version()

required_package = [
    'numpy >= 1.21.6',
    'scipy >= 1.7.3',
    'matplotlib >= 3.5.3',
    'psutil >= 5.9.4',
    'pandas >= 1.3.5',
    'scikit-learn >= 1.0.2',
    'lightgbm >= 3.3.5',
    'joblib >= 1.2.0',
    'seaborn >= 0.12.2',
    'pyyaml >= 6.0'
]

package_data = {
    '': [
        '*.so*',
        '*.pyd',
        'bin/*',
        'lib/*.so*',
        'lib/*.a',
        'include/*'
        'build_info.txt'
    ],
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
    "An AI framework for chemistry simulation",
    license='Apache 2.0',
    packages=find_packages(),
    package_data=package_data,
    include_package_data=True,
    install_requires=required_package,
    classifiers=['License :: OSI Approved :: Apache Software License'])
