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
import stat
from setuptools import setup
from setuptools import find_packages
from setuptools.command.egg_info import egg_info
from setuptools.command.build_py import build_py

cur_dir = os.path.dirname(os.path.realpath(__file__))
pkg_dir = os.path.join(cur_dir, 'build')
package_name = os.getenv('AICHEMIST_PACKAGE_NAME').replace("\n", "")


def read_version():
    """generate python file"""
    version_file = os.path.join(cur_dir, '../src/aichemist/', 'version.txt')
    with open(version_file, 'r') as f:
        version_ = f.readlines()[-1].strip()
    return version_

version = read_version()


def update_permissions(path):
    """
    Update permissions.

    Args:
        path (str): Target directory path.
    """
    for dirpath, dirnames, filenames in os.walk(path):
        for dirname in dirnames:
            dir_fullpath = os.path.join(dirpath, dirname)
            os.chmod(
                dir_fullpath, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC
                | stat.S_IRGRP | stat.S_IXGRP)
        for filename in filenames:
            file_fullpath = os.path.join(dirpath, filename)
            os.chmod(file_fullpath, stat.S_IREAD)


class EggInfo(egg_info):
    """Egg info."""
    def run(self):
        super().run()
        egg_info_dir = os.path.join(cur_dir, 'aichemist.egg-info')
        update_permissions(egg_info_dir)


class BuildPy(build_py):
    """BuildPy."""
    def run(self):
        super().run()
        aichemist_dir = os.path.join(pkg_dir, 'libs', 'aichemist')
        update_permissions(aichemist_dir)


with open('./requirements.txt', 'r') as f_requirements:
    requirements = f_requirements.readlines()
requirements = [r.strip() for r in requirements]
setup(name=package_name,
      version=version,
      author='Xuhan Liu',
      author_email='x@x',
      url='xxx',
      download_url='https://gitee.com/mindspore/mindscience/tags',
      project_urls={
          'Sources': 'https://gitee.com/mindspore/mindscience',
          'Issue Tracker': 'https://gitee.com/mindspore/mindscience/issues',
      },
      description=
      "Aichemist",
      license='Apache 2.0',
      packages=find_packages(),
      include_package_data=True,
      cmdclass={
          'egg_info': EggInfo,
          'build_py': BuildPy,
      },
      install_requires=requirements,
      classifiers=['License :: OSI Approved :: Apache Software License'])
