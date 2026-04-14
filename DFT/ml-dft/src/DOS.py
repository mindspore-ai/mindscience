# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ============================================================================
#
# Copyright 2022
# Georgia Tech Research Corporation
# Atlanta, Georgia 30332-4024
# All Rights Reserved
#
# This file is part of the ML-DFT originally developed by Georgia Tech Research Corporation (GTRC).
# Modifications have been made by Cen Jianhuan at Huawei Technologies Co., Ltd.
#
# This Program is licensed under the GENERAL PUBLIC USE LICENSE AGREEMENT provided by GTRC.
# You may not use this file except in compliance with the License.
#
# You should have received a copy of the GENERAL PUBLIC USE LICENSE AGREEMENT along with this program.
# If not, contact Georgia Tech Research Corporation for further information.
#
# Modifications made:
# - Replaced the neural network implementation originally based on Keras with an implementation using Huawei's MindSpore framework.
# - Completed training and inference on the Ascend NPU 910B.
#
# DISCLAIMER OF WARRANTIES AND LIMITATION ON LIABILITY:
# The Program is provided "AS IS", without warranty of any kind, express or implied, including but not limited to
# the warranties of merchantability, fitness for a particular purpose, and noninfringement. In no event shall the
# authors or copyright holders be liable for any claim, damages, or other liability, whether in an action of contract,
# tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software.
#
# ============================================================================
"""DOS implement"""

import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore.train import save_checkpoint
from mindspore import Tensor
from mindspore.dataset import GeneratorDataset


CONFIG_PATH = './checkpoints/weights_DOS.ckpt'

class single_atom_model(nn.Cell):
    def __init__(self, in_channels):
        super(single_atom_model, self).__init__()
        self.denseSeq = nn.SequentialCell(
            nn.Dense(in_channels, 600, activation='relu'),
            nn.Dropout(p=0.1),
            nn.Dense(600, 600, activation='relu'),
            nn.Dropout(p=0.1),
            nn.Dense(600, 600, activation='relu'),
            nn.Dropout(p=0.1),
            nn.Dense(600, 600, activation='relu'),
            nn.Dropout(p=0.1),
            nn.Dense(600, 343, activation='relu')
        )
        self.conv1 = nn.Conv1d(1, 3, 3,pad_mode="valid",has_bias=True)
        self.relu = nn.ReLU()

    def construct(self,modelC_in):
        modelC_out = self.denseSeq(modelC_in)
        modelC_out = ops.reshape(modelC_out,(-1,1,343))
        modelC_out = self.conv1(modelC_out)
        modelC_out = self.relu(modelC_out)
        modelC_out = ops.mean(modelC_out,axis=1)
        return modelC_out

class modelDOS(nn.Cell):
    def __init__(self,padding_size):
        super(modelDOS, self).__init__()
        self.time_distributed_1 = nn.TimeDistributed(single_atom_model(700),time_axis=1, reshape_with_axis=0)
        self.time_distributed_2 = nn.TimeDistributed(single_atom_model(568),time_axis=1, reshape_with_axis=0)
        self.time_distributed_3 = nn.TimeDistributed(single_atom_model(700),time_axis=1, reshape_with_axis=0)
        self.time_distributed_4 = nn.TimeDistributed(single_atom_model(700),time_axis=1, reshape_with_axis=0)
        self.dense1 = nn.Dense(341, 100, activation='relu')
        self.dense2 = nn.Dense(100, 100, activation='relu')
        self.dense3 = nn.Dense(100, 2,   activation='relu')
        self.padding_size = padding_size

    def construct(self,input1,input2,input3,input4,input5,input6,input7,input8,input9):
        model_out_C1 = self.time_distributed_1(input1)
        model_out_H1 = self.time_distributed_2(input2)
        model_out_N1 = self.time_distributed_3(input3)
        model_out_O1 = self.time_distributed_4(input4)
        D_C = ops.multiply(input6, model_out_C1)
        D_H = ops.multiply(input7, model_out_H1)
        D_N = ops.multiply(input8, model_out_N1)
        D_O = ops.multiply(input9, model_out_O1)
        model_added1 = ops.addn([D_C,D_H,D_N,D_O])

        model_s1 = ops.sum(model_added1,dim=1)
        model_dos = model_s1/input5
        bands = self.dense1(model_dos)
        bands = self.dense2(bands)
        bands = self.dense3(bands)

        return model_dos,bands

class init_DOSmod():
    def __init__(self,Padding_size):
        self._network = modelDOS(Padding_size)
        self._network.set_train(False)
        self.optim = nn.Adam(params=self._network.trainable_params(), learning_rate=0.0001,beta1=0.9, beta2=0.999, weight_decay=0.1)
        self.loss_fn = nn.MSELoss('mean')
        self.loss_weights = [1000,1]
    
    def predict(self, *inputs):
        return self._network(*inputs)
    
    def train(self, train_dataset, valid_dataset, epochs=100, filepath="newDOSmodel.ckpt"):
        self._network.set_train(True)
        min_err = 1e10

        # Define forward function
        def forward_fn(data, label):
            out1, out2 = self._network(*data)
            label1, label2 = label
            loss1 = self.loss_weights[0] * self.loss_fn(label1, out1) 
            loss2 = self.loss_weights[1] * self.loss_fn(label2, out2)
            loss = loss1 + loss2
            print("=====> loss1: ", loss1, "loss2: ", loss2)
            return loss, [out1, out2]

        # Get gradient function
        grad_fn = ms.value_and_grad(forward_fn, None, self.optim.parameters, has_aux=True)

        # Define function of one-step training
        def train_step(data, label):
            (loss, logits), grads = grad_fn(data, label)
            self.optim(grads)
            return loss, logits

        dataset_size = train_dataset.get_dataset_size()
        for epoch in range(epochs):
            self._network.set_train(True)
            for batch_idx, trainset in enumerate(train_dataset.create_dict_iterator()):
                X_C, X_H, X_N, X_O, X_el, C_d, H_d, N_d, O_d = trainset['X_C'], trainset['X_H'], trainset['X_N'], trainset['X_O'], trainset['X_el'], trainset['C_d'], trainset['H_d'], trainset['N_d'], trainset['O_d']
                Prop_dos, vbcb = trainset['Prop_dos'], trainset['vbcb']
                loss, logits = train_step([X_C,X_H,X_N,X_O,X_el,C_d,H_d,N_d,O_d],[Prop_dos,vbcb])

            for batch_idx, validset in enumerate(valid_dataset.create_dict_iterator()):
                X_val_C, X_val_H, X_val_N, X_val_O, X_el_val, C_dV, H_dV, N_dV, O_dV = validset['X_C'], validset['X_H'], validset['X_N'], validset['X_O'], validset['X_el'], validset['C_d'], validset['H_d'], validset['N_d'], validset['O_d']
                Prop_dos_val, vbcb_val = validset['Prop_dos'], validset['vbcb']
                err, _ = forward_fn([X_val_C, X_val_H, X_val_N, X_val_O, X_el_val, C_dV, H_dV, N_dV, O_dV], [Prop_dos_val, vbcb_val])
 
            print(f"Epoch: {epoch}, Loss: {loss.asnumpy()}, Error: {err.asnumpy()}")
            if err < min_err:
                min_err = err
                save_checkpoint(self._network, os.path.join(filepath))

def Dmodel_weights(train_dos,new_weights_dos,modelDOS):
    if train_dos:
        param_dict = ms.load_checkpoint("newDOSmodel.ckpt")
        ms.load_param_into_net(modelDOS._network, param_dict)
    elif new_weights_dos:
        param_dict = ms.load_checkpoint("newDOSmodel.ckpt")
        ms.load_param_into_net(modelDOS._network, param_dict)
    else:
        param_dict = ms.load_checkpoint(CONFIG_PATH)
        ms.load_param_into_net(modelDOS._network, param_dict)

def DOS_pred(X_C, X_H, X_N, X_O, total_elec, C_d, H_d, N_d, O_d, modelDOS, train_dos=False, new_weights_dos=False):
    resultD = []
    resultvbcb = []
    Dmodel_weights(train_dos, new_weights_dos, modelDOS)
    modelDOS._network.set_train(True)
    for i in range(100):
        modelInput = [X_C, X_H, X_N, X_O, total_elec, C_d, H_d, N_d, O_d]
        modelInput = [Tensor(inp, dtype=ms.float32) for inp in modelInput]
        Pred, vbcb = modelDOS.predict(*modelInput)
        Pred, vbcb = Pred.asnumpy(), vbcb.asnumpy()
        resultD.append(Pred * total_elec)
        resultvbcb.append(vbcb)
    resultD = np.array(resultD)
    Pred = resultD.mean(axis=0)
    resultvbcb = np.array(resultvbcb)
    devVB = resultvbcb.std(axis=0)[0][0]
    devCB = resultvbcb.std(axis=0)[0][1]
    Pred_vb = (-1) * resultvbcb.mean(axis=0)[0][0]
    Pred_cb = (-1) * resultvbcb.mean(axis=0)[0][1]
    uncertainty = resultD.std(axis=0)
    uncertainty = np.squeeze(uncertainty)
    resultvbcb = np.vstack(resultvbcb)
    Bandgap = Pred_cb - Pred_vb
    devBG = resultvbcb[:, 0] - resultvbcb[:, 1]
    devBG = devBG.std(axis=0)
    return Pred, uncertainty, Pred_vb, devVB, Pred_cb, devCB, Bandgap, devBG


def DOS_plot(energy_wind, Pred, VB, CB, uncertainty, localfile_loc, out_dir):
    plt.plot(energy_wind, Pred, "r-", label='DOS', linewidth=1)
    plt.fill_between(energy_wind, Pred - uncertainty, Pred + uncertainty, color='gray', alpha=0.2)
    plt.axvline(VB, color='b', label='Valence band', linestyle=':')
    plt.axvline(CB, color='g', label='Conduction band', linewidth=1)
    plt.axvline(0, color='k', label='Vacuum level', linestyle='dashed', linewidth=2)
    plt.legend(fontsize=16)
    plt.tick_params(labelsize=16)
    plt.xlabel("Energy (eV)", fontsize=18)
    plt.ylabel("DOS", fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "dos" + localfile_loc + ".png"), dpi=500)
    plt.clf()
    print("Made the dos plot ..")


def dos_data(file_loc, total_elec):
    dos_file = os.path.join(file_loc, "dos")
    dos_data = [[float(s) for s in l.split()] for l in open(dos_file).readlines()]
    dos = []
    for i in range(len(dos_data)):
        dos.append(dos_data[i][0])
    levels_file = os.path.join(file_loc, "VB_CB")
    levels_data = [[float(s) for s in l.split()] for l in open(levels_file).readlines()]
    VB = abs(np.array(levels_data[0]))
    CB = abs(np.array(levels_data[1]))
    Prop = np.array(dos) / total_elec
    return Prop, VB, CB

class DatasetGenerator:
    def __init__(self, X_C, X_H, X_N, X_O, X_el, C_d, H_d, N_d, O_d, Prop_dos, vbcb):
        self.data = {
            "X_C": np.array(X_C, dtype=np.float32),
            "X_H": np.array(X_H, dtype=np.float32),
            "X_N": np.array(X_N, dtype=np.float32),
            "X_O": np.array(X_O, dtype=np.float32),
            "X_el": np.array(X_el, dtype=np.float32),
            "C_d": np.array(C_d, dtype=np.float32),
            "H_d": np.array(H_d, dtype=np.float32),
            "N_d": np.array(N_d, dtype=np.float32),
            "O_d": np.array(O_d, dtype=np.float32),
            "Prop_dos": np.array(Prop_dos, dtype=np.float32),
            "vbcb": np.array(vbcb, dtype=np.float32)
        }

    def __getitem__(self, item):
        return self.data["X_C"][item], self.data["X_H"][item], self.data["X_N"][item], self.data["X_O"][item], self.data["X_el"][item], self.data["C_d"][item], self.data["H_d"][item], self.data["N_d"][item], self.data["O_d"][item], self.data["Prop_dos"][item], self.data["vbcb"][item]

    def __len__(self):
        return self.data["X_C"].shape[0]

def retrain_dosmodel(X_C, X_H, X_N, X_O, X_el, C_d, H_d, N_d, O_d, Prop_dos, vbcb, 
                     X_val_C, X_val_H, X_val_N, X_val_O, X_el_val, C_dV, H_dV, N_dV, O_dV, Prop_dos_val, vbcb_val, 
                     padding_size, config):
    drt_epochs, drt_batch_size, drt_patience = config["drt_epochs"], config["drt_batch_size"], config["drt_patience"]

    filepath = "newDOSmodel.ckpt"
    rtmodel = init_DOSmod(padding_size)
    param_dict = ms.load_checkpoint(CONFIG_PATH)
    ms.load_param_into_net(rtmodel._network, param_dict)

    train_dataset = DatasetGenerator(X_C, X_H, X_N, X_O, X_el, C_d, H_d, N_d, O_d, Prop_dos, vbcb)
    valid_dataset = DatasetGenerator(X_val_C, X_val_H, X_val_N, X_val_O, X_el_val, C_dV, H_dV, N_dV, O_dV, Prop_dos_val, vbcb_val)
    train_loader = GeneratorDataset(train_dataset, ["X_C", "X_H", "X_N", "X_O", "X_el", "C_d", "H_d", "N_d", "O_d", "Prop_dos", "vbcb"], shuffle=True)
    test_loader = GeneratorDataset(valid_dataset, ["X_C", "X_H", "X_N", "X_O", "X_el", "C_d", "H_d", "N_d", "O_d", "Prop_dos", "vbcb"], shuffle=True)

    train_loader = train_loader.batch(drt_batch_size)
    test_loader = test_loader.batch(drt_batch_size)

    steps_per_epoch = train_loader.get_dataset_size()
    rtmodel.train(train_loader, test_loader, epochs=drt_epochs, filepath=filepath)


