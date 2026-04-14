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
# ==============================================================================
"""convert HDF5 file to numpy"""
import os
import h5py
import numpy as np

PATH = "./sample"
LENGTH = 637
files = os.listdir(PATH)
num_png = len(files)
EZ_TOTAL = np.zeros([num_png, LENGTH])
HX_TOTAL = np.zeros([num_png, LENGTH])
HY_TOTAL = np.zeros([num_png, LENGTH])
NUMPOS = 2
LABEL_POS = np.zeros([num_png, NUMPOS])
pos_temp = np.zeros([1, NUMPOS])
STEP = 0
for filename in os.listdir(PATH):
    abspath = os.path.join(PATH, filename, './cylinder_Ascan_2D.out')
    f = h5py.File(abspath, 'r')
    Ez = f['/rxs/rx1/Ez'][:]
    Ez = Ez.reshape(1, LENGTH)
    Hx = f['/rxs/rx1/Hx'][:]
    Hx = Hx.reshape(1, LENGTH)
    Hy = f['/rxs/rx1/Hy'][:]
    Hy = Hy.reshape(1, LENGTH)
    EZ_TOTAL[STEP] = Ez
    HX_TOTAL[STEP] = Hx
    HY_TOTAL[STEP] = Hy
    temp_pos = filename.split("_")
    pos_temp[0][0] = temp_pos[-2]
    pos_temp[0][1] = temp_pos[-1]
    LABEL_POS[STEP] = pos_temp
    STEP += 1
np.save('Ez.npy', EZ_TOTAL)
np.save('Hx.npy', HX_TOTAL)
np.save('Hy.npy', HY_TOTAL)
np.save('Label.npy', LABEL_POS)
