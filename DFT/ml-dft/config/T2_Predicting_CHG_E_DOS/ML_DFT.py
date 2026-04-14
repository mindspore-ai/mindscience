import warnings
def fxn():
    warnings.warn("deprecated", DeprecationWarning)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
warnings.filterwarnings('ignore')

import numpy as np
import shutil
import pandas as pd
import os
import sys
sys.path.append(os.getcwd())
from inp_params import *
from sklearn.metrics import mean_absolute_error
import time
from pymatgen.io.vasp.outputs import Poscar
import os
import mindspore as ms

elec_dict={6:4, 1:1, 7:5, 8:6}
# package
from MLDFT.src.FP import fp_atom, fp_chg_norm, fp_norm
from MLDFT.src.CHG import init_chgmod, chg_predict, chg_ref, chg_pred_data, chg_pts,chg_print
from MLDFT.src.Energy import init_Emod, energy_predict, retrain_emodel
from MLDFT.src.DOS import init_DOSmod, DOS_pred, DOS_plot, retrain_dosmodel
from MLDFT.src.data_io import get_efp_data, dos_mask,get_e_dos_data, get_dos_data, get_dos_e_train_data, pad_efp_data, pad_dos_dat, pad_dat, chg_data

class Input_parameters:
    train_e=train_e
    ert_epochs=ert_epochs
    ert_batch_size=ert_batch_size
    ert_patience=ert_patience
    train_dos=train_dos
    drt_epochs=drt_epochs
    drt_batch_size=drt_batch_size
    drt_patience=drt_patience
    test_chg=test_chg
    tot_chg=tot_chg
    test_e=test_e
    test_dos=test_dos
    plot_dos=plot_dos
    write_chg=write_chg
    ref_chg=ref_chg
    grid_spacing=grid_spacing
    cut_off_rad = cut_off_rad
    batch_size_fp = batch_size_fp
    widest_gaussian = widest_gaussian
    narrowest_gaussian = narrowest_gaussian
    num_gamma = num_gamma
    fp_file=fp_file

inp_args=Input_parameters()
  
def ML_DFT(file_loc):
    poscar_file = os.path.join(file_loc,"POSCAR")
    poscar_data = Poscar.from_file(poscar_file)
    vol = poscar_data.structure.volume
    supercell = poscar_data.structure
    dim=supercell.lattice.matrix
    atoms=supercell.num_sites
    elems_list = sorted(list(set(poscar_data.site_symbols)))
    electrons_list = [elec_dict[x] for x in list(poscar_data.structure.atomic_numbers)]
    inp_args.total_elec = sum(electrons_list)
    dset,basis_mat,sites_elem,num_atoms,at_elem=fp_atom(poscar_data,supercell,elems_list)
    total_elec=inp_args.total_elec
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
        
    localfile_loc = file_loc.replace("/", "_")
    print("Writing atomic charges to text files...")
    np.savetxt("C_charges" + localfile_loc + ".txt",np.c_[C_at_charge])
    np.savetxt("H_charges" + localfile_loc + ".txt",np.c_[H_at_charge])
    if i3!= 0:
        np.savetxt("N_charges" + localfile_loc + ".txt",np.c_[N_at_charge])
    if i4!=0:
        np.savetxt("O_charges" + localfile_loc + ".txt",np.c_[O_at_charge])

    if test_e or test_dos:
        X_C,X_H,X_N,X_O=fp_chg_norm(Coef_at1,Coef_at2,Coef_at3,Coef_at4,X_3D1,X_3D2,X_3D3,X_3D4,padding_size)
    if test_e:
        modelE=init_Emod(padding_size)
        Pred_Energy,ForC,ForH,ForN,ForO,Stress=energy_predict(X_C,X_H,X_N,X_O,basis1,basis2,basis3,basis4,C_m,H_m,N_m,O_m,num_atoms.reshape(1,1),modelE)
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
        Pred, uncertainty,VB,devVB,CB,devCB,BG,devBG=DOS_pred(X_C,X_H,X_N,X_O,np.array(total_elec).reshape(1,1),C_d,H_d,N_d,O_d,modelD)
        DOS=np.squeeze(Pred)
        print('Valence band maximum:', VB, '+-', devVB, ' eV')
        print('Conduction band minimum:', CB, '+-', devCB, ' eV')
        print('Bandgap:', BG, '+-', devBG, ' eV')
        energy_wind=np.arange(-33.0,1.1,0.1)
        print("Writing DOS curve to text file...")
        np.savetxt("DOS" + localfile_loc + ".txt",np.c_[energy_wind,DOS])
        if plot_dos:
            DOS_plot(energy_wind,DOS,VB,CB,uncertainty,localfile_loc)
    if comp_chg:
        shutil.copy2(poscar_file, "Pred_CHG_test"+ localfile_loc +".dat")
        chg_coor,chg_den,num_pts=chg_ref(file_loc,vol, supercell)
        Pred_chg=chg_pred_data(poscar_data,at_elem,sites_elem,Coef_at1,Coef_at2,Coef_at3,Coef_at4,chg_coor,dim,vol,tot_chg)
        ae_chg=mean_absolute_error(chg_den,Pred_chg)*len(chg_den)
        dft_chg=np.sum(chg_den)
        comp=100*(ae_chg/dft_chg)
        print("Predicted charge error (%):", comp)
    if write_chg:
        shutil.copy2(poscar_file, "Pred_CHG_test"+ localfile_loc +".dat")
        if ref_chg:
            chg_coor,chg_den,num_pts=chg_ref(file_loc,vol, supercell)    
        else:
            chg_coor,num_pts=chg_pts(poscar_data, supercell,grid_spacing)
        if not comp_chg:
            Pred_chg=chg_pred_data(poscar_data,at_elem,sites_elem,Coef_at1,Coef_at2,Coef_at3,Coef_at4,chg_coor,dim,vol,tot_chg)
        chg_print(Pred_chg,vol,localfile_loc,num_pts)


if __name__ == '__main__':    
    ms.set_context(mode=ms.context.PYNATIVE_MODE, device_target="GPU", device_id=7)
    orig_stdout = sys.stdout
    f = open('OUT_DATA.log', 'w')
    sys.stdout = f

    start_time = time.time()
    if train_e or train_dos:
        df_train= pd.read_csv("Train.csv")
        df_val=pd.read_csv("Val.csv")
        train_list=df_train['files']
        val_list=df_val['files']

    if not train_dos and train_e: 
        ener_ref,forces_pre_list,press_ref,X_pre_list,basis_pre_list,X_at,X_el,X_elem=get_efp_data(train_list)
        ener_val,forcesV_pre_list,press_val,XV_pre_list,basisV_pre_list,X_at_val,X_el_val,X_elem_val=get_efp_data(val_list)
        padding_size=max(np.amax(X_elem),np.amax(X_elem_val))
        forces1,forces2,forces3,forces4,X_1,X_2,X_3,X_4,basis1,basis2,basis3,basis4,C_m,H_m,N_m,O_m=pad_efp_data(X_elem,X_pre_list,forces_pre_list,basis_pre_list,padding_size)
        forcesV1,forcesV2,forcesV3,forcesV4,X_1V,X_2V,X_3V,X_4V,basis1V,basis2V,basis3V,basis4V,C_mV,H_mV,N_mV,O_mV=pad_efp_data(X_elem_val,XV_pre_list,forcesV_pre_list,basisV_pre_list,padding_size)
        modelCHG=init_chgmod(padding_size)
        X_C,X_H,X_N,X_O=get_dos_e_train_data(X_1,X_2,X_3,X_4,X_elem,padding_size,modelCHG)
        X_val_C,X_val_H,X_val_N,X_val_O=get_dos_e_train_data(X_1V,X_2V,X_3V,X_4V,X_elem_val,padding_size,modelCHG)
        X_C,X_H,X_N,X_O=fp_norm(X_C,X_H,X_N,X_O,padding_size)
        X_val_C,X_val_H,X_val_N,X_val_O=fp_norm(X_val_C,X_val_H,X_val_N,X_val_O,padding_size)
        retrain_emodel(X_C,X_H,X_N,X_O,C_m,H_m,N_m,O_m,basis1,basis2,basis3,basis4,X_at,ener_ref,forces1,forces2,forces3,forces4,press_ref,X_val_C,X_val_H,X_val_N,X_val_O,C_mV,H_mV,N_mV,O_mV,basis1V,basis2V,basis3V,basis4V,X_at_val,ener_val,forcesV1,forcesV2,forcesV3,forcesV4,press_val,ert_epochs,ert_batch_size, ert_patience,padding_size)

    if train_e and train_dos:
        ener_ref,forces_pre_list,press_ref,X_pre_list,basis_pre_list,X_at,X_el,X_elem,Prop_dos,Prop_vbcb=get_e_dos_data(train_list)
        ener_val,forcesV_pre_list,press_val,XV_pre_list,basisV_pre_list,X_at_val,X_el_val,X_elem_val,Prop_dos_val,Prop_vbcb_val=get_e_dos_data(val_list)    
        padding_size=max(np.amax(X_elem),np.amax(X_elem_val))
        forces1,forces2,forces3,forces4,X_1,X_2,X_3,X_4,basis1,basis2,basis3,basis4,C_m,H_m,N_m,O_m=pad_efp_data(X_elem,X_pre_list,forces_pre_list,basis_pre_list,padding_size)
        forcesV1,forcesV2,forcesV3,forcesV4,X_1V,X_2V,X_3V,X_4V,basis1V,basis2V,basis3V,basis4V,C_mV,H_mV,N_mV,O_mV=pad_efp_data(X_elem_val,XV_pre_list,forcesV_pre_list,basisV_pre_list,padding_size)
        vbcb,C_d,H_d,N_d,O_d=pad_dos_dat(Prop_vbcb,X_1,C_m,H_m,N_m,O_m,padding_size)
        vbcb_val,C_dV,H_dV,N_dV,O_dV=pad_dos_dat(Prop_vbcb_val,X_1V,C_mV,H_mV,N_mV,O_mV,padding_size)
        modelCHG=init_chgmod(padding_size)
        X_C,X_H,X_N,X_O=get_dos_e_train_data(X_1,X_2,X_3,X_4,X_elem,padding_size,modelCHG)
        X_val_C,X_val_H,X_val_N,X_val_O=get_dos_e_train_data(X_1V,X_2V,X_3V,X_4V,X_elem_val,padding_size,modelCHG)
        X_C,X_H,X_N,X_O=fp_norm(X_C,X_H,X_N,X_O,padding_size)
        X_val_C,X_val_H,X_val_N,X_val_O=fp_norm(X_val_C,X_val_H,X_val_N,X_val_O,padding_size)
        retrain_emodel(X_C,X_H,X_N,X_O,C_m,H_m,N_m,O_m,basis1,basis2,basis3,basis4,X_at,ener_ref,forces1,forces2,forces3,forces4,press_ref,X_val_C,X_val_H,X_val_N,X_val_O,C_mV,H_mV,N_mV,O_mV,basis1V,basis2V,basis3V,basis4V,X_at_val,ener_val,forcesV1,forcesV2,forcesV3,forcesV4,press_val,ert_epochs,ert_batch_size, ert_patience,padding_size)
        retrain_dosmodel(X_C,X_H,X_N,X_O,X_el,C_d,H_d,N_d,O_d,Prop_dos,vbcb,X_val_C,X_val_H,X_val_N,X_val_O,X_el_val,C_dV,H_dV,N_dV,O_dV,Prop_dos_val,vbcb_val,drt_epochs,drt_batch_size, drt_patience,padding_size)

    if not train_e and train_dos:
        X_pre_list,X_at,X_el,X_elem,Prop_dos,Prop_vbcb=get_dos_data(train_list)
        XV_pre_list,X_at_val,X_el_val,X_elem_val,Prop_dos_val,Prop_vbcb_val=get_dos_data(val_list)
        padding_size=max(np.amax(X_elem),np.amax(X_elem_val))
        X_1,X_2,X_3,X_4,C_m,H_m,N_m,O_m=pad_dat(X_elem,X_pre_list,padding_size)
        X_1V,X_2V,X_3V,X_4V,C_mV,H_mV,N_mV,O_mV=pad_dat(X_elem_val,XV_pre_list,padding_size)
        vbcb,C_d,H_d,N_d,O_d=pad_dos_dat(Prop_vbcb,X_1,C_m,H_m,N_m,O_m,padding_size)
        vbcb_val,C_dV,H_dV,N_dV,O_dV=pad_dos_dat(Prop_vbcb_val,X_1V,C_mV,H_mV,N_mV,O_mV,padding_size)
        modelCHG=init_chgmod(padding_size)
        X_C,X_H,X_N,X_O=get_dos_e_train_data(X_1,X_2,X_3,X_4,X_elem,padding_size,modelCHG)
        X_val_C,X_val_H,X_val_N,X_val_O=get_dos_e_train_data(X_1V,X_2V,X_3V,X_4V,X_elem_val,padding_size,modelCHG)
        X_C,X_H,X_N,X_O=fp_norm(X_C,X_H,X_N,X_O,padding_size)
        X_val_C,X_val_H,X_val_N,X_val_O=fp_norm(X_val_C,X_val_H,X_val_N,X_val_O,padding_size)
        retrain_dosmodel(X_C,X_H,X_N,X_O,X_el,C_d,H_d,N_d,O_d,Prop_dos,vbcb,X_val_C,X_val_H,X_val_N,X_val_O,X_el_val,C_dV,H_dV,N_dV,O_dV,Prop_dos_val,vbcb_val,drt_epochs,drt_batch_size, drt_patience,padding_size)

    train_time = time.time()

    if test_chg or test_e or test_dos :
        df_test = pd.read_csv("predict.csv")
        file_loc_test = df_test['file_loc_test'].tolist()
        inp_args.file_loc_test= [x for x in file_loc_test  if str(x) != 'nan']
        for file_loc in inp_args.file_loc_test:
            dirname=file_loc
            print('file:', dirname)
            ML_DFT(dirname)

    inference_time = time.time()

    print('====================Total Training Time: {} s'.format(train_time-start_time))
    print('====================Total Inference Time: {} s'.format(inference_time-train_time))
    print('====================Total Time: {} s'.format(inference_time-start_time))

    sys.stdout = orig_stdout
    f.close()
