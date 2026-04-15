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
"""xcnn Config"""
import configparser


def _get_typed_arg(f, key, fallback=None, tolist=False):
    v = f(key, fallback=fallback)
    if v.__class__.__name__ == 'unicode': v = str(v)
    if tolist: v = v.split()
    return v


def parse_xcnn_options(sec):
    xcnn_opts = dict()
    xcnn_opts['Verbose'] = _get_typed_arg(sec.getboolean, 'Verbose', fallback=True)
    xcnn_opts['CheckPointPath'] = _get_typed_arg(sec.get, 'CheckPointPath', fallback='.')
    xcnn_opts['EnableCuda'] = _get_typed_arg(sec.getboolean, 'EnableCuda', fallback=True)

    xcnn_opts['Structure'] = _get_typed_arg(sec.get, 'Structure')
    xcnn_opts['OrbitalBasis'] = _get_typed_arg(sec.get, 'OrbitalBasis')
    xcnn_opts['ReferencePotential'] = _get_typed_arg(sec.get, 'ReferencePotential', fallback='hfx', tolist=True)

    xcnn_opts['Model'] = _get_typed_arg(sec.get, 'Model')
    xcnn_opts['ModelPath'] = _get_typed_arg(sec.get, 'ModelPath')

    xcnn_opts['MeshLevel'] = _get_typed_arg(sec.getint, 'MeshLevel', fallback=3)
    xcnn_opts['CubeLength'] = _get_typed_arg(sec.getfloat, 'CubeLength', fallback=0.9)
    xcnn_opts['CubePoint'] = _get_typed_arg(sec.getint, 'CubePoint', fallback=9)
    xcnn_opts['Symmetric'] = _get_typed_arg(sec.get, 'Symmetric', fallback=None)

    xcnn_opts['InitDensityMatrix'] = _get_typed_arg(sec.get, 'InitDensityMatrix', fallback='rks') 
    xcnn_opts['xcFunctional'] = _get_typed_arg(sec.get, 'xcFunctional', fallback='b3lypg')
    xcnn_opts['ConvergenceCriterion'] = _get_typed_arg(sec.getfloat, 'ConvergenceCriterion', fallback=1e-6)
    xcnn_opts['MaxIteration'] = _get_typed_arg(sec.getint, 'MaxIteration', fallback=99)

    xcnn_opts['ZeroForceConstrain'] = _get_typed_arg(sec.getboolean, 'ZeroForceConstrain', fallback=True)
    xcnn_opts['ZeroTorqueConstrain'] = _get_typed_arg(sec.getboolean, 'ZeroTorqueConstrain', fallback=False)

    return xcnn_opts


def get_options(config_file, sec):
    config = configparser.ConfigParser()
    config.read(config_file)
    return parse_xcnn_options(config['XCNN'])


