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
"""Energy implement"""

import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore.train import save_checkpoint
from mindspore.dataset import GeneratorDataset
import os


CONFIG_PATH = './checkpoints/weights_EFP.ckpt'

class single_atom_model_E(nn.Cell):
    def __init__(self, in_channels):
        super(single_atom_model_E, self).__init__()
        self.denseSeq = nn.SequentialCell(
            nn.Dense(in_channels, 300, activation='tanh'),
            nn.Dense(300, 300, activation='tanh'),
            nn.Dense(300, 300, activation='tanh'),
            nn.Dense(300, 300, activation='tanh'),
            nn.Dense(300, 300, activation='tanh'),
            nn.Dense(300, 10)
        )
        self.in_channels = in_channels

    def construct(self,model_input):
        model_out, basis = ops.split(model_input,[self.in_channels,9], axis=-1)
        basis = ops.reshape(basis,(-1,3,3))
        Tbasis = ops.transpose(basis,(0,2,1))

        model_out = self.denseSeq(model_out)
        E, forces, XX, YY, ZZ, XY, YZ, XZ = ops.split(model_out,[1,3,1,1,1,1,1,1],axis=-1)
        Energy = ops.abs(E)
        forces = ops.reshape(forces,(-1,3,1))
        forces_out = ops.matmul(basis, forces)
        forces_out = ops.reshape(forces_out,(-1,3))
        model_out = ops.concat((XX,XY,XZ,XY,YY,YZ,XZ,YZ,ZZ),axis=-1)
        model_out = ops.reshape(model_out, (-1, 3, 3))
        model_out = ops.matmul(basis, model_out)
        model_out = ops.matmul(model_out,Tbasis)
        model_out = ops.reshape(model_out,(-1,9))
        XX,XY,XZ,YX,YY,YZ,ZX,ZY,ZZ = ops.split(model_out, [1,1,1,1,1,1,1,1,1], axis=-1)
        model_out = ops.concat([Energy,forces_out,XX,YY,ZZ,XY,YZ,XZ],axis=-1)
        return model_out

class modelEmod(nn.Cell):
    def __init__(self,padding_size):
        super(modelEmod, self).__init__()
        self.time_distributed_1 = nn.TimeDistributed(single_atom_model_E(700),time_axis=1, reshape_with_axis=0)
        self.time_distributed_2 = nn.TimeDistributed(single_atom_model_E(568),time_axis=1, reshape_with_axis=0)
        self.time_distributed_3 = nn.TimeDistributed(single_atom_model_E(700),time_axis=1, reshape_with_axis=0)
        self.time_distributed_4 = nn.TimeDistributed(single_atom_model_E(700),time_axis=1, reshape_with_axis=0)
        self.padding_size = padding_size

    def construct(self,input1,input2,input3,input4,input5,input6,input7,input8,input9):
        model_out_CP = self.time_distributed_1(input1)
        model_out_HP = self.time_distributed_2(input2)
        model_out_NP = self.time_distributed_3(input3)
        model_out_OP = self.time_distributed_4(input4)
        EC,forcesC,pressC = ops.split(model_out_CP, [1,3,6], axis=-1)
        EH,forcesH,pressH = ops.split(model_out_HP, [1,3,6], axis=-1)
        EN,forcesN,pressN = ops.split(model_out_NP, [1,3,6], axis=-1)
        EO,forcesO,pressO = ops.split(model_out_OP, [1,3,6], axis=-1)
        model_added = ops.addn([pressC, pressH, pressN, pressO])
        EC = ops.multiply(input6, EC)
        EH = ops.multiply(input7, EH)
        EN = ops.multiply(input8, EN)
        EO = ops.multiply(input9, EO)
        E_added = ops.addn([EC, EH, EN, EO])
        E_tot = ops.sum(E_added, dim=1)
        E_tot = E_tot / input5
        model_p = ops.sum(model_added, dim=1)
        model_p = model_p / input5
        return E_tot,forcesC,forcesH,forcesN,forcesO,model_p
    
class se(nn.Cell):
    def __init__(self):
        super(se, self).__init__()

    def construct(self, y_pred, y_true):
        mask = ops.all(ops.equal(y_true, 1000), axis=-1, keep_dims=True)
        mask = 1 - ops.cast(mask, ms.float32)
        diff = ops.subtract(y_true,y_pred)
        loss = ops.square(diff)*mask
        loss = ops.sum(loss)
        cases= ops.sum(mask)
        loss = ops.divide(loss,cases)
        return loss

class init_Emod():
    def __init__(self,Padding_size):
        self._network = modelEmod(Padding_size)
        self._network.set_train(False)
        self.optim = nn.Adam(params=self._network.trainable_params(), learning_rate=0.00005,beta1=0.9, beta2=0.999)
        self.loss_mse = nn.MSELoss()
        self.loss_se = se()
        self.value_and_grad = ms.value_and_grad(self.forward_fn, None, self.optim.parameters, has_aux=True)
        self.loss_weights = [1000,10,10,10,10,0.1]

    def predict(self, *inputs):
        return self._network(*inputs)

    # Define forward function
    def forward_fn(self, data, label):
        out1, out2, out3, out4, out5, out6 = self._network(*data)
        label1, label2, label3, label4, label5, label6 = label
        loss1 = self.loss_weights[0] * self.loss_mse(out1, label1)
        loss2 = self.loss_weights[1] * self.loss_se(out2, label2)
        loss3 = self.loss_weights[2] * self.loss_se(out3, label3)
        loss4 = self.loss_weights[3] * self.loss_se(out4, label4)
        loss5 = self.loss_weights[4] * self.loss_se(out5, label5)
        loss6 = self.loss_weights[5] * self.loss_mse(out6, label6)
        loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
        return loss, [out1, out2, out3, out4, out5, out6]
    # Define function of one-step training
    def train_step(self, data, label):
        (loss, logits), grads = self.value_and_grad(data, label)
        self.optim(grads)
        return loss, logits
    
    def train(self, train_dataset, valid_dataset, epochs=100, filepath="newEmodel.ckpt"):
        self._network.set_train(True)
        min_err = 1e10

        dataset_size = train_dataset.get_dataset_size()
        for epoch in range(epochs):
            for batch_idx, trainset in enumerate(train_dataset.create_dict_iterator()):
                X_C,X_H,X_N,X_O,X_at,C_m,H_m,N_m,O_m= trainset['X_C'], trainset['X_H'], trainset['X_N'], trainset['X_O'], trainset['X_at'], trainset['C_m'], trainset['H_m'], trainset['N_m'], trainset['O_m']
                ener_ref,forces1,forces2,forces3,forces4,press_ref = trainset['ener_ref'], trainset['forces1'], trainset['forces2'], trainset['forces3'], trainset['forces4'], trainset['press_ref']
                loss, logits = self.train_step([X_C,X_H,X_N,X_O,X_at,C_m,H_m,N_m,O_m], [ener_ref,forces1,forces2,forces3,forces4,press_ref])

            self._network.set_train(False)
            for batch_idx, validset in enumerate(valid_dataset.create_dict_iterator()):
                X_val_C,X_val_H,X_val_N,X_val_O,X_at_val,C_mV,H_mV,N_mV,O_mV= validset['X_C'], validset['X_H'], validset['X_N'], validset['X_O'], validset['X_at'], validset['C_m'], validset['H_m'], validset['N_m'], validset['O_m']
                ener_val,forces1V,forces2V,forces3V,forces4V,press_val = validset['ener_ref'], validset['forces1'], validset['forces2'], validset['forces3'], validset['forces4'], validset['press_ref']
                err, logits = self.forward_fn([X_val_C,X_val_H,X_val_N,X_val_O,X_at_val,C_mV,H_mV,N_mV,O_mV], [ener_val,forces1V,forces2V,forces3V,forces4V,press_val])
            self._network.set_train(True)

            print(f"Epoch: {epoch}, Loss: {loss.asnumpy()}, Error: {err.asnumpy()}")
            if err < min_err:
                min_err = err
                save_checkpoint(self._network, os.path.join(filepath))

def model_weights(train_e,new_weights_e,model_E):
    if train_e:
        param_dict = ms.load_checkpoint("newEmodel.ckpt")
        ms.load_param_into_net(model_E._network, param_dict)
    elif new_weights_e:
        param_dict = ms.load_checkpoint("newEmodel.ckpt")
        ms.load_param_into_net(model_E._network, param_dict)
    else:
        param_dict = ms.load_checkpoint(CONFIG_PATH)
        ms.load_param_into_net(model_E._network, param_dict)


def energy_predict(X_C,X_H,X_N,X_O,basis1,basis2,basis3,basis4,C_m,H_m,N_m,O_m,num_atoms,model_E,train_e=False,new_weights_e=False):
    X_C=np.concatenate((X_C,basis1), axis=-1)
    X_H=np.concatenate((X_H,basis2), axis=-1)
    X_N=np.concatenate((X_N,basis3), axis=-1)
    X_O=np.concatenate((X_O,basis4), axis=-1)
    model_weights(train_e,new_weights_e,model_E)
    inputs = [X_C,X_H,X_N,X_O,num_atoms,C_m,H_m,N_m,O_m]
    inputs = [ms.Tensor(i,dtype=ms.float32) for i in inputs]
    E,ForC,ForH,ForN,ForO,pred_press=model_E.predict(*inputs)
    E,ForC,ForH,ForN,ForO,pred_press=E.asnumpy(),ForC.asnumpy(),ForH.asnumpy(),ForN.asnumpy(),ForO.asnumpy(),pred_press.asnumpy()
    Pred_Energy=(-1)*E
    ForC=np.squeeze(ForC)
    ForH=np.squeeze(ForH)
    ForN=np.squeeze(ForN)
    ForO=np.squeeze(ForO)
    pred_press=np.squeeze(pred_press)
    return Pred_Energy[0][0],ForC,ForH,ForN,ForO,pred_press

def e_train(file_loc,tot_atoms):
    levels_file=os.path.join(file_loc,"energy")
    levels_data=[[float(s) for s in l.split()] for l in open(levels_file).readlines()]
    Energy=abs(np.array(levels_data[0]))/tot_atoms
    forces_file=os.path.join(file_loc,"forces")
    forces_data=np.array([[float(s) for s in l.split()] for l in open(forces_file).readlines()])
    press_file=os.path.join(file_loc,"stress")
    press_data=np.array([[float(s) for s in l.split()] for l in open(press_file).readlines()])
    press_data=np.reshape(press_data,(6))
    press=np.reshape(press_data,(1,6))

    return Energy,forces_data,press

class DatasetGenerator:
    def __init__(self, X_C,X_H,X_N,X_O,X_at,C_m,H_m,N_m,O_m,ener_ref,forces1,forces2,forces3,forces4,press_ref):
        self.data = {
            "X_C": np.array(X_C, dtype=np.float32),
            "X_H": np.array(X_H, dtype=np.float32),
            "X_N": np.array(X_N, dtype=np.float32),
            "X_O": np.array(X_O, dtype=np.float32),
            "X_at": np.array(X_at, dtype=np.float32),
            "C_m": np.array(C_m, dtype=np.float32),
            "H_m": np.array(H_m, dtype=np.float32),
            "N_m": np.array(N_m, dtype=np.float32),
            "O_m": np.array(O_m, dtype=np.float32),
            "ener_ref": np.array(ener_ref, dtype=np.float32),
            "forces1": np.array(forces1, dtype=np.float32),
            "forces2": np.array(forces2, dtype=np.float32),
            "forces3": np.array(forces3, dtype=np.float32),
            "forces4": np.array(forces4, dtype=np.float32),
            "press_ref": np.array(press_ref, dtype=np.float32)
        }

    def __getitem__(self, item):
        return self.data["X_C"][item], self.data["X_H"][item], self.data["X_N"][item], self.data["X_O"][item], self.data["X_at"][item], self.data["C_m"][item], self.data["H_m"][item], self.data["N_m"][item], self.data["O_m"][item], self.data["ener_ref"][item], self.data["forces1"][item], self.data["forces2"][item], self.data["forces3"][item], self.data["forces4"][item], self.data["press_ref"][item]

    def __len__(self):
        return self.data["X_C"].shape[0]
    

def retrain_emodel(X_C,X_H,X_N,X_O,C_m,H_m,N_m,O_m,basis1,basis2,basis3,basis4,X_at,ener_ref,forces1,forces2,forces3,forces4,press_ref,
                    X_val_C,X_val_H,X_val_N,X_val_O,C_mV,H_mV,N_mV,O_mV,basis1V,basis2V,basis3V,basis4V,X_at_val,ener_val,forces1V,forces2V,forces3V,forces4V,press_val,
                    padding_size,config):
    ert_epochs, ert_batch_size, ert_patience = config["ert_epochs"], config["ert_batch_size"], config["ert_patience"]

    X_C=np.concatenate((X_C,basis1), axis=-1)
    X_H=np.concatenate((X_H,basis2), axis=-1)
    X_N=np.concatenate((X_N,basis3), axis=-1)
    X_O=np.concatenate((X_O,basis4), axis=-1)
    X_val_C=np.concatenate((X_val_C,basis1V), axis=-1)
    X_val_H=np.concatenate((X_val_H,basis2V), axis=-1)
    X_val_N=np.concatenate((X_val_N,basis3V), axis=-1)
    X_val_O=np.concatenate((X_val_O,basis4V), axis=-1)

    filepath="newEmodel.ckpt"
    rtmodel = init_Emod(padding_size)
    param_dict = ms.load_checkpoint(CONFIG_PATH)
    ms.load_param_into_net(rtmodel._network, param_dict)

    train_dataset = DatasetGenerator(X_C, X_H, X_N, X_O, X_at, C_m, H_m, N_m, O_m, ener_ref, forces1, forces2, forces3, forces4, press_ref)
    val_dataset = DatasetGenerator(X_val_C, X_val_H, X_val_N, X_val_O, X_at_val, C_mV, H_mV, N_mV, O_mV, ener_val, forces1V, forces2V, forces3V, forces4V, press_val)
    train_loader = GeneratorDataset(train_dataset, ["X_C", "X_H", "X_N", "X_O", "X_at", "C_m", "H_m", "N_m", "O_m", "ener_ref", "forces1", "forces2", "forces3", "forces4", "press_ref"], shuffle=True)
    test_loader = GeneratorDataset(val_dataset, ["X_C", "X_H", "X_N", "X_O", "X_at", "C_m", "H_m", "N_m", "O_m", "ener_ref", "forces1", "forces2", "forces3", "forces4", "press_ref"], shuffle=True)

    train_loader = train_loader.batch(ert_batch_size)
    test_loader = test_loader.batch(ert_batch_size)

    steps_per_epoch = train_loader.get_dataset_size()
    rtmodel.train(train_loader, test_loader, epochs=ert_epochs, filepath=filepath)