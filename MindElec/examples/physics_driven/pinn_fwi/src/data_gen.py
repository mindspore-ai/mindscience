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
"""dataset"""

import os
import yaml
from SALib.sample import sobol_sequence
import scipy.interpolate as interpolate
import numpy as np
import mindspore.dataset as ds
import mindspore as ms

with open('src/default_config.yaml', 'r') as y:
    cfg = yaml.full_load(y)
# 定义训练数据 包括内部采样和初始条件及边界采样
dx = cfg['ax_spec']/cfg['nx']
dz = cfg['az_spec']/cfg['nz']
ax = cfg['xsf']-cfg['n_absx']*dx
az = cfg['az_spec']-cfg['n_absz']*dz

t01 = 2000*cfg['s_spec']  # 第一个快照
t02 = 2300*cfg['s_spec']  # 第二个快照
t_la = 5000*cfg['s_spec']

zl_s = 0.06-cfg['n_absz']*dz
z0_s = az

if cfg['is_Train']:
    n_pde = cfg['batch_size']*cfg['batch_number']
    print('batch_size', ':', cfg['batch_size'])
    x_pde = sobol_sequence.sample(n_pde+1, 3)[1:, :]
    x_pde[:, 0] = x_pde[:, 0] * ax/cfg['Lx']
    x_pde[:, 1] = x_pde[:, 1] * az/cfg['Lz']
    x_pde[:, 2] = x_pde[:, 2] * (cfg['t_m']-cfg['t_st'])


X0 = np.loadtxt('%s/wavefields/wavefield_grid_for_dumps_000.txt' %
                (cfg['data_dir']))  # 监视器位置
X0 = X0/1000  # specfem得到的数据单位是m，将其转成Km
X0[:, 0:1] = X0[:, 0:1]/cfg['Lx']
X0[:, 1:2] = X0[:, 1:2]/cfg['Lz']
xz = np.concatenate((X0[:, 0:1], X0[:, 1:2]), axis=1)


ININUM = 40
xx, zz = np.meshgrid(np.linspace(
    0, ax/cfg['Lx'], ININUM), np.linspace(0, az/cfg['Lz'], ININUM))
xxzz = np.concatenate((xx.reshape((-1, 1)), zz.reshape((-1, 1))), axis=1)
x_init1 = np.concatenate((xx.reshape((-1, 1)), zz.reshape((-1, 1)),
                          0.0*np.ones((ININUM**2, 1), dtype=np.float64)), axis=1)
x_init2 = np.concatenate((xx.reshape((-1, 1)), zz.reshape((-1, 1)),
                          (t02-t01)*np.ones((ININUM**2, 1), dtype=np.float64)), axis=1)

# 吸收条件相关
xf = cfg['n_absx']*dx
zf = cfg['n_absz']*dz
xxs, zzs = np.meshgrid(np.linspace(
    xf/cfg['Lx'], cfg['xsf']/cfg['Lx'], ININUM), np.linspace(zf/cfg['Lz'], cfg['az_spec']/cfg['Lz'], ININUM))
xxzzs = np.concatenate((xxs.reshape((-1, 1)), zzs.reshape((-1, 1))), axis=1)


USCL = 1/3640  # scaling the output data to cover [-1 1] interval


wfs = sorted(os.listdir('%s/wavefields/.' % (cfg['data_dir'])))
U0 = [np.loadtxt('%s/wavefields/' % (cfg['data_dir'])+f) for f in wfs]

u_ini1 = interpolate.griddata(xz, U0[0], xxzzs, fill_value=0.0)
u_ini1x = u_ini1[:, 0:1]/USCL
u_ini1z = u_ini1[:, 1:2]/USCL


u_ini2 = interpolate.griddata(xz, U0[1], xxzzs, fill_value=0.0)
u_ini2x = u_ini2[:, 0:1]/USCL
u_ini2z = u_ini2[:, 1:2]/USCL

u_spec = interpolate.griddata(xz, U0[2], xxzzs, fill_value=0.0)  # Test data
u_specx = u_spec[:, 0:1]/USCL
u_specz = u_spec[:, 1:2]/USCL


for ii in list(range(cfg['n_event']-1)):
    wfs = sorted(os.listdir('event'+str(ii+2)+'/wavefields/.'))
    U0 = [np.loadtxt('event'+str(ii+2)+'/wavefields/'+f) for f in wfs]

    u_ini1 = interpolate.griddata(xz, U0[0], xxzzs, fill_value=0.0)
    u_ini1x += u_ini1[:, 0:1]/USCL
    u_ini1z += u_ini1[:, 1:2]/USCL

    u_ini2 = interpolate.griddata(xz, U0[1], xxzzs, fill_value=0.0)
    u_ini2x += u_ini2[:, 0:1]/USCL
    u_ini2z += u_ini2[:, 1:2]/USCL

    u_spec = interpolate.griddata(xz, U0[2], xxzzs, fill_value=0.0)
    u_specx += u_spec[:, 0:1]/USCL
    u_specz += u_spec[:, 1:2]/USCL


sms = sorted(os.listdir('%s/seismograms/.' % (cfg['data_dir'])))
smsz = [f for f in sms if f[-6] == 'Z']
seismo_listz = [np.loadtxt('%s/seismograms/' %
                           (cfg['data_dir'])+f) for f in smsz]

t_spec = -seismo_listz[0][0, 0]+seismo_listz[0][:, 0]
cut_u = t_spec > cfg['t_s']
cut_l = t_spec < cfg['t_st']
l_su = len(cut_u)-sum(cut_u)
l_sl = sum(cut_l)


LF = 100
index = np.arange(l_sl, l_su, LF)
l_sub = len(index)
t_spec_sub = t_spec[index].reshape((-1, 1))

t_spec_sub = t_spec_sub-t_spec_sub[0]


for ii in list(range(len(seismo_listz))):
    seismo_listz[ii] = seismo_listz[ii][index]


s_z = seismo_listz[0][:, 1].reshape(-1, 1)
for ii in list(range(len(seismo_listz)-1)):
    s_z = np.concatenate((s_z, seismo_listz[ii+1][:, 1].reshape(-1, 1)), axis=0)


for ii in list(range(cfg['n_event']-1)):
    sms = sorted(os.listdir('event'+str(ii+2)+'/seismograms/.'))
    smsz = [f for f in sms if f[-6] == 'Z']
    seismo_listz = [np.loadtxt(
        'event'+str(ii+2)+'/seismograms/'+f) for f in smsz]

    for jj in list(range(len(seismo_listz))):
        seismo_listz[jj] = seismo_listz[jj][index]

    Sze = seismo_listz[0][:, 1].reshape(-1, 1)
    for jj in list(range(len(seismo_listz)-1)):
        Sze = np.concatenate(
            (Sze, seismo_listz[jj+1][:, 1].reshape(-1, 1)), axis=0)

    s_z += Sze


s_z = s_z/USCL

X_S = np.empty([int(np.size(s_z)), 3])
d_s = np.abs((zl_s-z0_s))/(cfg['n_seis']-1)

for ii in list(range(len(seismo_listz))):
    X_S[ii*l_sub:(ii+1)*l_sub, :] = np.concatenate((ax/cfg['Lx']*np.ones(
        (l_sub, 1), dtype=np.float64), (z0_s-ii*d_s)/cfg['Lz']*np.ones(
            (l_sub, 1), dtype=np.float64), t_spec_sub), axis=1)


sms = sorted(os.listdir('%s/seismograms/.' % (cfg['data_dir'])))
smsx = [f for f in sms if f[-6] == 'X']
seismo_listx = [np.loadtxt('%s/seismograms/' %
                           (cfg['data_dir'])+f) for f in smsx]


for ii in list(range(len(seismo_listx))):
    seismo_listx[ii] = seismo_listx[ii][index]


s_x = seismo_listx[0][:, 1].reshape(-1, 1)
for ii in list(range(len(seismo_listx)-1)):
    s_x = np.concatenate((s_x, seismo_listx[ii+1][:, 1].reshape(-1, 1)), axis=0)


for ii in list(range(cfg['n_event']-1)):
    sms = sorted(os.listdir('event'+str(ii+2)+'/seismograms/.'))
    smsx = [f for f in sms if f[-6] == 'X']  # X cmp seismos
    seismo_listx = [np.loadtxt(
        'event'+str(ii+2)+'/seismograms/'+f) for f in smsx]

    for jj in list(range(len(seismo_listx))):
        seismo_listx[jj] = seismo_listx[jj][index]

    sxe = seismo_listx[0][:, 1].reshape(-1, 1)
    for jj in list(range(len(seismo_listx)-1)):
        sxe = np.concatenate(
            (sxe, seismo_listx[jj+1][:, 1].reshape(-1, 1)), axis=0)

    s_x += sxe


s_x = s_x/USCL

####  边界条件 ####

BCXNUM = 100
BCTNUM = 50
x_vec = np.random.rand(BCXNUM, 1)*ax/cfg['Lx']
t_vec = np.random.rand(BCTNUM, 1)*(cfg['t_m']-cfg['t_st'])
xxb, ttb = np.meshgrid(x_vec, t_vec)
x_bc_t = np.concatenate((xxb.reshape((-1, 1)), az/cfg['Lz']*np.ones(
    (xxb.reshape((-1, 1)).shape[0], 1)), ttb.reshape((-1, 1))), axis=1)


class DatasetGenerator:
    """generate training dataset"""

    def __init__(self):

        data = np.empty((0, 3))
        for i in list(range(cfg['batch_number'])):
            # 针对方程和边界条件，分别定义一个新的训练的batch
            self.x_vec = np.random.rand(BCXNUM, 1)*ax/cfg['Lx']
            self.t_vec = np.random.rand(BCTNUM, 1)*(cfg['t_m']-cfg['t_st'])
            self.xxb, self.ttb = np.meshgrid(self.x_vec, self.t_vec)
            self.x_bc_t = np.concatenate((self.xxb.reshape((-1, 1)), az/cfg['Lz']*np.ones(
                (self.xxb.reshape((-1, 1)).shape[0], 1)), self.ttb.reshape((-1, 1))), axis=1)
            data = np.concatenate((data, x_pde[i*cfg['batch_size']:(
                i+1)*cfg['batch_size']], x_init1, x_init2, X_S, self.x_bc_t), axis=0)

        self.data = data

    def __getitem__(self, idx):

        return self.data[idx]

    def __len__(self):

        return len(self.data)


if cfg['is_Train']:

    N1 = cfg['batch_size']
    N2 = x_init1.shape[0]
    N3 = x_init2.shape[0]
    N4 = X_S.shape[0]
    N5 = x_bc_t.shape[0]

    # 实例化数据集
    dataset_generator = DatasetGenerator()
    dataset = ds.GeneratorDataset(dataset_generator, ["data"], shuffle=False)
    train_dataset = dataset.batch(N1+N2+N3+N4+N5)

xx0, zz0 = xx.reshape((-1, 1)), zz.reshape((-1, 1))

# evaluating PINNs at time=0
x_eval01 = np.concatenate((xx0, zz0, 0*np.ones((xx0.shape[0], 1))), axis=1)
# evaluating PINNs at time when the second input from specfem is provided
x_eval02 = np.concatenate(
    (xx0, zz0, (t02-t01)*np.ones((xx0.shape[0], 1))), axis=1)
# evaluating PINNs at a later time>0
x_evalt = np.concatenate(
    (xx0, zz0, (t_la-t01)*np.ones((xx0.shape[0], 1))), axis=1)

dataset01 = ms.Tensor(x_eval01, ms.float32)
dataset02 = ms.Tensor(x_eval02, ms.float32)
dataset2 = ms.Tensor(x_evalt, ms.float32)
dataset_seism = ms.Tensor(X_S, ms.float32)
