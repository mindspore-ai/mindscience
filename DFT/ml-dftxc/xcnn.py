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
"""xcnn"""
from __future__ import print_function
import os, sys, errno
import argparse
import numpy as np
import mindspore as ms
import pyscf
ver = pyscf.__version__.split('.')
import configparser
from mindspore import context, set_seed

from src.xcnn.Config import get_options
from src.xcnn.nnks import NNKS
from src.xcnn.utility import dft, ccsd, I

set_seed(123456)
np.random.seed(123456)


def parse_args():
    '''Parse input args'''
    parser = argparse.ArgumentParser(description="xcnn")
    parser.add_argument("--config", type=str, default="./config/test.cfg")
    parser.add_argument("--device_target", type=str, default="Ascend", choices=["GPU", "Ascend", "CPU"],
                        help="The target device to run, support 'Ascend', 'GPU', 'CPU")
    parser.add_argument("--device_id", type=int, default=0, help="ID of the target device")
    parser.add_argument("--mode", type=str, default="GRAPH", choices=["GRAPH", "PYNATIVE"],
                        help="Running in GRAPH_MODE OR PYNATIVE_MODE")
    input_args = parser.parse_args()
    return input_args

def main():
    args = parse_args()
    config = configparser.ConfigParser()
    config.read(args.config)
    sections = config.sections()

    xcnn_opts = get_options(args.config, 'XCNN')
    for k, v in xcnn_opts.items():
        print(k, '\t', v, '\t', v.__class__)

    context.set_context(mode=context.GRAPH_MODE if args.mode.upper().startswith("GRAPH") else context.PYNATIVE_MODE,
                        device_target=args.device_target,
                        device_id=args.device_id)
    print(f"Running in {args.mode.upper()} mode, using device id: {args.device_id}.")
    
    print('Check point files saved to %s' % (xcnn_opts['CheckPointPath']))
    try: 
        os.makedirs(xcnn_opts['CheckPointPath'])
    #except FileExistsError:
    except OSError as e:
        if e.errno == errno.EEXIST:
            print('CHK folder already exists. Old files will be overwritten.')

    nnks = NNKS(xcnn_opts)
    print('Prepare to perform post DFT SCF calculation.')
    nnks.scf()
    np.save('%s/dm_nn' % (xcnn_opts['CheckPointPath']), nnks.dm)
    dm_rks = dft(nnks.mol, xc='B3LYPG')
    dm_cc = ccsd(nnks.mol)
    np.save('%s/dm_rks_b3lypg' % (xcnn_opts['CheckPointPath']), dm_rks)
    np.save('%s/dm_ccsd' % (xcnn_opts['CheckPointPath']), dm_cc)

    print()
    print('---- Force ----')
    force = nnks.force(True)
    print('Force on atoms')
    for f in force:
        print('%+16.12e %+16.12e %+16.12e' % (f[0], f[1], f[2]))
    print()

    mtorque = np.cross(force, nnks.mol.atom_coords())
    print('mtorque')
    for t in mtorque:
        print('%+16.12e %+16.12e %+16.12e' % (t[0], t[1], t[2]))
    print()

    print('---- I ----')
    print('rks, nn vs ccsd: %16.12e\t%16.12e' % 
            (
                I(nnks.mol, dm_cc, dm_rks, nnks.grids._coords, nnks.grids.weights), 
                I(nnks.mol, dm_cc, nnks.dm, nnks.grids._coords, nnks.grids.weights), 
            ))


if __name__ == '__main__':
    print("pid:", os.getpid())
    main()

