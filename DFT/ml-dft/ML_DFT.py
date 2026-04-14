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
"""main"""

import warnings
def fxn():
    warnings.warn("deprecated", DeprecationWarning)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
warnings.filterwarnings('ignore')

import os
import time
import argparse
import shutil
import sys
sys.path.append(os.getcwd())

import yaml
import numpy as np
import pandas as pd
from   pymatgen.io.vasp.outputs import Poscar
from   sklearn.metrics import mean_absolute_error

import mindspore as ms
from   mindspore import context, nn, ops, jit, set_seed

from src.CHG import init_chgmod, chg_predict, chg_ref, chg_pred_data, chg_pts, chg_print
from src.Energy import init_Emod, energy_predict, retrain_emodel
from src.DOS import init_DOSmod, DOS_pred, DOS_plot, retrain_dosmodel
from src.dataset import get_efp_data, dos_mask,get_e_dos_data, get_dos_data, get_dos_e_train_data, pad_efp_data, pad_dos_dat, pad_dat, chg_data
from src.utils import fp_atom, fp_chg_norm, fp_norm

# Initialize the random seed
set_seed(123456)
np.random.seed(123456)
# atom types and their corresponding number of electrons
elec_dict = {6:4, 1:1, 7:5, 8:6}

def parse_args():
    '''Parse input arguments'''
    parser = argparse.ArgumentParser(description='ML-DFT for predicting atomic charges, energy, forces, and DOS')
    parser.add_argument('--config', help='config file', default='./config/T1_config.yaml')
    parser.add_argument('--device_target', type=str, default='Ascend', choices=['Ascend', 'GPU', 'CPU'],
                        help='The target device to run, support "Ascend", "GPU", "CPU". Default: "Ascend".')
    parser.add_argument('--device_id', type=int, default=1, help='Device id, default is 1.')
    parser.add_argument('--mode', type=str, default='pynative', choices=['pynative', 'graph'],
                        help='Running in PYNATIVE_MODE or GRAPH_MODE, default is pynative.')
    input_args = parser.parse_args()
    return input_args

def retrain(config):
    train_e    = config['train_e']
    train_dos  = config['train_dos']

    df_train   = pd.read_csv(os.path.join(config['data_path'], "Train.csv"))
    df_val     = pd.read_csv(os.path.join(config['data_path'], "Val.csv"))
    train_list = df_train['files']
    val_list   = df_val['files']

    if train_e and train_dos:
        # Get the data for training the energy model and DOS model
        ener_ref,forces_pre_list,press_ref,X_pre_list,basis_pre_list,X_at,X_el,X_elem,Prop_dos,Prop_vbcb=get_e_dos_data(train_list)
        ener_val,forcesV_pre_list,press_val,XV_pre_list,basisV_pre_list,XV_at,XV_el,XV_elem,Prop_dosV,Prop_vbcbV=get_e_dos_data(val_list)  
        # Pad the data to the same size  
        padding_size=max(np.amax(X_elem),np.amax(XV_elem))
        forces1,forces2,forces3,forces4,X_1,X_2,X_3,X_4,basis1,basis2,basis3,basis4,C_m,H_m,N_m,O_m=pad_efp_data(X_elem,X_pre_list,forces_pre_list,basis_pre_list,padding_size)
        forcesV1,forcesV2,forcesV3,forcesV4,X_1V,X_2V,X_3V,X_4V,basis1V,basis2V,basis3V,basis4V,C_mV,H_mV,N_mV,O_mV=pad_efp_data(XV_elem,XV_pre_list,forcesV_pre_list,basisV_pre_list,padding_size)
        vbcb,C_d,H_d,N_d,O_d=pad_dos_dat(Prop_vbcb,X_1,C_m,H_m,N_m,O_m,padding_size)
        vbcbV,C_dV,H_dV,N_dV,O_dV=pad_dos_dat(Prop_vbcbV,X_1V,C_mV,H_mV,N_mV,O_mV,padding_size)
        modelCHG=init_chgmod(padding_size)
        X_C,X_H,X_N,X_O=get_dos_e_train_data(X_1,X_2,X_3,X_4,X_elem,padding_size,modelCHG)
        XV_C,XV_H,XV_N,XV_O=get_dos_e_train_data(X_1V,X_2V,X_3V,X_4V,XV_elem,padding_size,modelCHG)
        X_C,X_H,X_N,X_O=fp_norm(X_C,X_H,X_N,X_O,padding_size)
        XV_C,XV_H,XV_N,XV_O=fp_norm(XV_C,XV_H,XV_N,XV_O,padding_size)
        retrain_emodel(X_C,X_H,X_N,X_O,C_m,H_m,N_m,O_m,basis1,basis2,basis3,basis4,X_at,ener_ref,forces1,forces2,forces3,forces4,press_ref,
                       XV_C,XV_H,XV_N,XV_O,C_mV,H_mV,N_mV,O_mV,basis1V,basis2V,basis3V,basis4V,XV_at,ener_val,forcesV1,forcesV2,forcesV3,forcesV4,press_val,
                       padding_size,config)
        retrain_dosmodel(X_C,X_H,X_N,X_O,X_el,C_d,H_d,N_d,O_d,Prop_dos,vbcb,XV_C,XV_H,XV_N,XV_O,XV_el,C_dV,H_dV,N_dV,O_dV,Prop_dosV,vbcbV,padding_size,config)
    elif train_e:
        # Get the data for training the energy model
        ener_ref, forces_pre_list,  press_ref, X_pre_list,  basis_pre_list,  X_at,  X_el,  X_elem  = get_efp_data(train_list)
        ener_val, forcesV_pre_list, press_val, XV_pre_list, basisV_pre_list, XV_at, XV_el, XV_elem = get_efp_data(val_list)
        padding_size = max(np.amax(X_elem), np.amax(XV_elem))
        forces1,  forces2,  forces3,  forces4,  X_1,  X_2,  X_3,  X_4,  basis1,  basis2,  basis3,  basis4,  C_m,  H_m,  N_m,  O_m  = pad_efp_data(X_elem,  X_pre_list,  forces_pre_list,  basis_pre_list,  padding_size)
        forcesV1, forcesV2, forcesV3, forcesV4, X_1V, X_2V, X_3V, X_4V, basis1V, basis2V, basis3V, basis4V, C_mV, H_mV, N_mV, O_mV = pad_efp_data(XV_elem, XV_pre_list, forcesV_pre_list, basisV_pre_list, padding_size)
        modelCHG = init_chgmod(padding_size)
        X_C,  X_H,  X_N,  X_O  = get_dos_e_train_data(X_1,X_2,X_3,X_4,X_elem,padding_size,modelCHG)
        XV_C, XV_H, XV_N, XV_O = get_dos_e_train_data(X_1V,X_2V,X_3V,X_4V,XV_elem,padding_size,modelCHG)
        X_C,  X_H,  X_N,  X_O  = fp_norm(X_C,X_H,X_N,X_O,padding_size)
        XV_C, XV_H, XV_N, XV_O = fp_norm(XV_C,XV_H,XV_N,XV_O,padding_size)
        retrain_emodel(X_C,X_H,X_N,X_O,C_m,H_m,N_m,O_m,basis1,basis2,basis3,basis4,X_at,ener_ref,forces1,forces2,forces3,forces4,press_ref,
                       XV_C,XV_H,XV_N,XV_O,C_mV,H_mV,N_mV,O_mV,basis1V,basis2V,basis3V,basis4V,XV_at,ener_val,forcesV1,forcesV2,forcesV3,forcesV4,press_val,
                       padding_size,config)
    elif train_dos:
        # Get the data for training the DOS model
        X_pre_list,  X_at,  X_el,  X_elem,  Prop_dos,  Prop_vbcb  = get_dos_data(train_list)
        XV_pre_list, XV_at, XV_el, XV_elem, Prop_dosV, Prop_vbcbV = get_dos_data(val_list)
        padding_size = max(np.amax(X_elem), np.amax(XV_elem))
        X_1,  X_2,  X_3,  X_4,  C_m,  H_m,  N_m,  O_m  = pad_dat(X_elem,X_pre_list,padding_size)
        X_1V, X_2V, X_3V, X_4V, C_mV, H_mV, N_mV, O_mV = pad_dat(XV_elem,XV_pre_list,padding_size)
        vbcb,  C_d,  H_d,  N_d,  O_d  = pad_dos_dat(Prop_vbcb,X_1,C_m,H_m,N_m,O_m,padding_size)
        vbcbV, C_dV, H_dV, N_dV, O_dV = pad_dos_dat(Prop_vbcbV,X_1V,C_mV,H_mV,N_mV,O_mV,padding_size)
        modelCHG = init_chgmod(padding_size)
        X_C,X_H,X_N,X_O=get_dos_e_train_data(X_1,X_2,X_3,X_4,X_elem,padding_size,modelCHG)
        XV_C,XV_H,XV_N,XV_O=get_dos_e_train_data(X_1V,X_2V,X_3V,X_4V,XV_elem,padding_size,modelCHG)
        X_C,X_H,X_N,X_O=fp_norm(X_C,X_H,X_N,X_O,padding_size)
        XV_C,XV_H,XV_N,XV_O=fp_norm(XV_C,XV_H,XV_N,XV_O,padding_size)
        retrain_dosmodel(X_C,X_H,X_N,X_O,X_el,C_d,H_d,N_d,O_d,Prop_dos,vbcb,XV_C,XV_H,XV_N,XV_O,XV_el,C_dV,H_dV,N_dV,O_dV,Prop_dosV,vbcbV,padding_size,config)

  
def ML_DFT(file_loc, config):
    test_e, test_dos, plot_dos, write_chg, ref_chg, comp_chg = config['test_e'], config['test_dos'], config['plot_dos'], config['write_chg'], config['ref_chg'], config['comp_chg']
    grid_spacing, tot_chg = config['grid_spacing'], config['tot_chg']
    out_dir  = config['output_path']
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    poscar_file = os.path.join(file_loc,"POSCAR")
    poscar_data = Poscar.from_file(poscar_file)
    vol = poscar_data.structure.volume
    supercell = poscar_data.structure
    dim=supercell.lattice.matrix
    atoms=supercell.num_sites
    elems_list = sorted(list(set(poscar_data.site_symbols)))
    electrons_list = [elec_dict[x] for x in list(poscar_data.structure.atomic_numbers)]
    total_elec = sum(electrons_list)
    dset,basis_mat,sites_elem,num_atoms,at_elem=fp_atom(poscar_data,supercell,elems_list)
    dataset1 = dset[:]

    print('Total number of electrons inside cell:',total_elec)
    i1=at_elem[0]
    i2=at_elem[1]
    i3=at_elem[2]
    i4=at_elem[3]
    padding_size=max([i1,i2,i3,i4])
    num_atoms=np.array(dataset1.shape[0])

    X_3D1,X_3D2,X_3D3,X_3D4,basis1,basis2,basis3,basis4,C_m,H_m,N_m,O_m=chg_data(dataset1,basis_mat,i1,i2,i3,i4,padding_size)
    modelCHG=init_chgmod(padding_size)
    Coef_at1,Coef_at2,Coef_at3, Coef_at4,C_at_charge, H_at_charge, N_at_charge, O_at_charge=chg_predict(X_3D1,X_3D2,X_3D3,X_3D4,i1,i2,i3,i4,sites_elem,modelCHG,at_elem)
    print('Atomic charges for the C atoms (same order as in POSCAR):', C_at_charge)
    print('Atomic charges for the H atoms (same order as in POSCAR):', H_at_charge)
    if i3!= 0:
        print('Atomic charges for the N atoms (same order as in POSCAR):', N_at_charge)
    if i4!=0:
        print('Atomic charges for the O atoms (same order as in POSCAR):', O_at_charge)
        
    localfile_loc = file_loc.replace(".", "").replace("/", "_")
    print("Writing atomic charges to text files...")
    np.savetxt(os.path.join(out_dir, "C_charges" + localfile_loc + ".txt"), np.c_[C_at_charge])
    np.savetxt(os.path.join(out_dir, "H_charges" + localfile_loc + ".txt"), np.c_[H_at_charge])
    if i3!= 0:
        np.savetxt(os.path.join(out_dir, "N_charges" + localfile_loc + ".txt"), np.c_[N_at_charge])
    if i4!=0:
        np.savetxt(os.path.join(out_dir, "O_charges" + localfile_loc + ".txt"), np.c_[O_at_charge])

    if test_e or test_dos:
        X_C,X_H,X_N,X_O=fp_chg_norm(Coef_at1,Coef_at2,Coef_at3,Coef_at4,X_3D1,X_3D2,X_3D3,X_3D4,padding_size)
    if test_e:
        modelE=init_Emod(padding_size)
        Pred_Energy,ForC,ForH,ForN,ForO,Stress=energy_predict(X_C,X_H,X_N,X_O,basis1,basis2,basis3,basis4,C_m,H_m,N_m,O_m,num_atoms.reshape(1,1),modelE,config['train_e'],config['new_weights_e'])
        Forces=np.concatenate((ForC[0:i1],ForH[0:i2]),axis=0)
        if i3!= 0:
            Forces=np.concatenate((Forces,ForN[0:i3]),axis=0)
        if i4!= 0:
            Forces=np.concatenate((Forces,ForO[0:i4]),axis=0)
        print('Total potential energy:', Pred_Energy*num_atoms, ' eV')
        print('Atomic forces (eV/A):', Forces)
        print('The stress tensor components are (kB): Sxx:', Stress[0],' Syy:', Stress[1],' Szz:', Stress[2], ' Sxy:', Stress[3],' Syz:', Stress[4],' Sxz:', Stress[5] )
    if test_dos:
        modelD=init_DOSmod(padding_size)
        C_d,H_d,N_d,O_d=dos_mask(C_m,H_m,N_m,O_m,padding_size)
        Pred, uncertainty,VB,devVB,CB,devCB,BG,devBG=DOS_pred(X_C,X_H,X_N,X_O,np.array(total_elec).reshape(1,1),C_d,H_d,N_d,O_d,modelD,config['train_dos'],config['new_weights_dos'])
        DOS=np.squeeze(Pred)
        print('Valence band maximum:', VB, '+-', devVB, ' eV')
        print('Conduction band minimum:', CB, '+-', devCB, ' eV')
        print('Bandgap:', BG, '+-', devBG, ' eV')
        energy_wind=np.arange(-33.0,1.1,0.1)
        print("Writing DOS curve to text file...")
        np.savetxt(os.path.join(out_dir, "DOS" + localfile_loc + ".txt"), np.c_[energy_wind, DOS])
        if plot_dos:
            DOS_plot(energy_wind,DOS,VB,CB,uncertainty,localfile_loc,out_dir)
    if comp_chg:
        shutil.copy2(poscar_file, os.path.join(out_dir, "Pred_CHG_test"+ localfile_loc +".dat"))
        chg_coor,chg_den,num_pts=chg_ref(file_loc,vol, supercell)
        Pred_chg=chg_pred_data(poscar_data,at_elem,sites_elem,Coef_at1,Coef_at2,Coef_at3,Coef_at4,chg_coor,dim,vol,tot_chg)
        ae_chg=mean_absolute_error(chg_den,Pred_chg)*len(chg_den)
        dft_chg=np.sum(chg_den)
        comp=ae_chg/dft_chg
        print("Predicted charge error:", comp)
    if write_chg:
        shutil.copy2(poscar_file, os.path.join(out_dir, "Pred_CHG_test"+ localfile_loc +".dat"))
        if ref_chg:
            chg_coor,chg_den,num_pts=chg_ref(file_loc,vol, supercell)    
        else:
            chg_coor,num_pts=chg_pts(poscar_data, supercell,grid_spacing)
        if not comp_chg:
            Pred_chg=chg_pred_data(poscar_data,at_elem,sites_elem,Coef_at1,Coef_at2,Coef_at3,Coef_at4,chg_coor,dim,vol,tot_chg)
        chg_print(Pred_chg,vol,localfile_loc,num_pts,out_dir)


if __name__ == '__main__':    
    print("pid:", os.getpid())
    args = parse_args()
    config = yaml.load(open(args.config), Loader=yaml.FullLoader)
    context.set_context(mode=context.GRAPH_MODE if args.mode.upper().startswith('GRAPH') else context.PYNATIVE_MODE, 
                        device_target=args.device_target, 
                        device_id=args.device_id)
    print(f"Running in {args.mode.upper()} mode, using device {args.device_target}, device id {args.device_id}")
    use_ascend = context.get_context(attr_key="device_target") == "Ascend"

    start_time = time.time()

    if config['train_e'] or config['train_dos']:
        retrain(config)
    train_time = time.time()

    if config['test_chg'] or config['test_e'] or config['test_dos']:
        df_test = pd.read_csv(os.path.join(config['data_path'], "predict.csv"))
        file_loc_test = df_test['file_loc_test'].tolist()
        file_loc_test = [x for x in file_loc_test if str(x) != 'nan']
        for dirname in file_loc_test:
            print('file:', dirname)
            ML_DFT(dirname, config)
    inference_time = time.time()

    print("Training cost {} s".format(train_time-start_time))
    print("Inference cost {} s".format(inference_time-train_time))
    print("Total cost {} s".format(inference_time-start_time))
