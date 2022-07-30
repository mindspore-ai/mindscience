#include "main_ti.cuh"

CONTROLLER controller;
TI_CORE TI_core;
BOND bondA;
BOND bondB;
BOND_SOFT bond_soft;
ANGLE angleA;
ANGLE angleB;
DIHEDRAL dihedralA;
DIHEDRAL dihedralB;
NON_BOND_14 nb14A;
NON_BOND_14 nb14B;
NEIGHBOR_LIST neighbor_list;
LJ_SOFT_CORE lj_soft;
Particle_Mesh_Ewald pme;

int main(int argc, char *argv[])
{
    Main_Initial(argc, argv);
    for (TI_core.input.current_frame = 1; TI_core.input.current_frame <= TI_core.input.frame_numbers; ++TI_core.input.current_frame)
    {
        Main_Calculation();
        Main_Print();
        if (TI_core.input.current_frame != TI_core.input.frame_numbers)
        {
            Main_Iteration();
        }
    }
    TI_core.Print_dH_dlambda_Average_To_Screen_And_Result_File();
    Main_Clear();
    return 0;
}

void Main_Initial(int argc, char *argv[])
{
    controller.Initial(argc, argv);
    TI_core.Initial(&controller);

    
    nb14A.Initial(&controller, NULL, NULL, NULL, "nb14A");
    nb14B.Initial(&controller, NULL, NULL, NULL, "nb14B");
    bondA.Initial(&controller, "bondA");
    bondB.Initial(&controller, "bondB");
    bond_soft.Initial(&controller);
    angleA.Initial(&controller, "angleA");
    angleB.Initial(&controller, "angleB");
    dihedralA.Initial(&controller, "dihedralA");
    dihedralB.Initial(&controller, "dihedralB");


    lj_soft.Initial(&controller, TI_core.nb.cutoff, TI_core.box_length);
    if (TI_core.charge_pertubated)
    {
        pme.Initial(&controller, TI_core.atom_numbers, TI_core.box_length, TI_core.nb.cutoff);
        TI_core.cross_pme.Initial(TI_core.atom_numbers, pme.PME_Nall);
    }

    if (TI_core.charge_pertubated || lj_soft.is_initialized)
    {
        neighbor_list.Initial(&controller, TI_core.atom_numbers, TI_core.box_length, TI_core.nb.cutoff, TI_core.nb.skin);
        neighbor_list.Neighbor_List_Update(TI_core.crd, TI_core.nb.d_excluded_list_start, TI_core.nb.d_excluded_list, TI_core.nb.d_excluded_numbers, 1);
    }

    controller.Input_Check();
    controller.Print_First_Line_To_Mdout();
}

void Main_Calculation()
{
    TI_core.data.dH_dlambda_current_frame = 0.0;
    if (bondA.is_initialized)
        TI_core.data.bondA_ene = bondA.Get_Energy(TI_core.uint_crd, TI_core.pbc.uint_dr_to_dr_cof);
    else
        TI_core.data.bondA_ene = 0.0;

    if (bondB.is_initialized)
        TI_core.data.bondB_ene = bondB.Get_Energy(TI_core.uint_crd, TI_core.pbc.uint_dr_to_dr_cof);
    else
        TI_core.data.bondB_ene = 0.0;
    
    if (angleA.is_initialized)
        TI_core.data.angleA_ene = angleA.Get_Energy(TI_core.uint_crd, TI_core.pbc.uint_dr_to_dr_cof);
    else
        TI_core.data.angleA_ene = 0.0;

    if (angleB.is_initialized)
        TI_core.data.angleB_ene = angleB.Get_Energy(TI_core.uint_crd, TI_core.pbc.uint_dr_to_dr_cof);
    else
        TI_core.data.angleB_ene = 0.0;

    if (dihedralA.is_initialized)
        TI_core.data.dihedralA_ene = dihedralA.Get_Energy(TI_core.uint_crd, TI_core.pbc.uint_dr_to_dr_cof);
    else
        TI_core.data.dihedralA_ene = 0.0;

    if (dihedralB.is_initialized)
        TI_core.data.dihedralB_ene = dihedralB.Get_Energy(TI_core.uint_crd, TI_core.pbc.uint_dr_to_dr_cof);
    else
        TI_core.data.dihedralB_ene = 0.0;

     if (nb14A.is_initialized)
    {
        TI_core.data.nb14A_EE_ene = nb14A.Get_14_CF_Energy(TI_core.uint_crd, TI_core.d_charge, TI_core.pbc.uint_dr_to_dr_cof);
        TI_core.data.nb14A_LJ_ene = nb14A.Get_14_LJ_Energy(TI_core.uint_crd, TI_core.pbc.uint_dr_to_dr_cof);
    }
    else
    {
        TI_core.data.nb14A_EE_ene = 0.0;
        TI_core.data.nb14A_LJ_ene = 0.0;
    }

    if (nb14B.is_initialized)
    {
        TI_core.data.nb14B_EE_ene = nb14B.Get_14_CF_Energy(TI_core.uint_crd, TI_core.d_charge, TI_core.pbc.uint_dr_to_dr_cof);
        TI_core.data.nb14B_LJ_ene = nb14B.Get_14_LJ_Energy(TI_core.uint_crd, TI_core.pbc.uint_dr_to_dr_cof);
    }
    else
    {
        TI_core.data.nb14B_EE_ene = 0.0;
        TI_core.data.nb14B_LJ_ene = 0.0;
    }

    if (bond_soft.is_initialized)
        TI_core.data.bond_soft_dH_dlambda = bond_soft.Get_Partial_H_Partial_Lambda(TI_core.uint_crd, TI_core.pbc.uint_dr_to_dr_cof);
    else
        TI_core.data.bond_soft_dH_dlambda = 0.0;

    if (lj_soft.is_initialized)
    {
        TI_core.data.lj_soft_dH_dlambda = lj_soft.Get_Partial_H_Partial_Lambda_With_Columb_Direct(TI_core.uint_crd, TI_core.d_charge, neighbor_list.d_nl, TI_core.d_charge_B_A, TI_core.charge_pertubated);
        TI_core.data.coul_direct_dH_dlambda = *lj_soft.h_sigma_of_dH_dlambda_direct;
        TI_core.data.lj_soft_long_range_correction = lj_soft.Partial_H_Partial_Lambda_Long_Range_Correction();
    }
    else
    {
        TI_core.data.lj_soft_dH_dlambda = 0.0;
        TI_core.data.coul_direct_dH_dlambda = 0.0;
        TI_core.data.lj_soft_long_range_correction = 0.0;
    }


    if (TI_core.charge_pertubated)
    {
        TI_core.data.pme_dH_dlambda = TI_core.Get_Cross_PME_Partial_H_Partial_Lambda(&pme, neighbor_list.d_nl, lj_soft.is_initialized);
        TI_core.data.pme_self_dH_dlambda = TI_core.cross_pme.cross_self_ene;
        TI_core.data.pme_reci_dH_dlambda = TI_core.cross_pme.cross_reciprocal_ene;
        TI_core.data.pme_corr_dH_dlambda = TI_core.cross_pme.cross_correction_ene;
        TI_core.data.coul_direct_dH_dlambda += TI_core.cross_pme.cross_direct_ene;
    }
    else
    {
        TI_core.data.pme_self_dH_dlambda = 0.0f;
        TI_core.data.pme_reci_dH_dlambda = 0.0f;
        TI_core.data.pme_corr_dH_dlambda = 0.0f;
    }
    
    TI_core.data.Sum_One_Frame();
}

void Main_Print()
{
    controller.Step_Print("frame", TI_core.input.current_frame);
    controller.Step_Print("dH_dlambda", TI_core.data.dH_dlambda_current_frame);
    controller.Step_Print("bondA", TI_core.data.bondA_ene);
    controller.Step_Print("bondB", TI_core.data.bondB_ene);
    controller.Step_Print("angleA", TI_core.data.angleA_ene);
    controller.Step_Print("angleB", TI_core.data.angleB_ene);
    controller.Step_Print("dihedralA", TI_core.data.dihedralA_ene);
    controller.Step_Print("dihedralB", TI_core.data.dihedralB_ene);
    controller.Step_Print("nb14A_LJ", TI_core.data.nb14A_LJ_ene);
    controller.Step_Print("nb14B_LJ", TI_core.data.nb14B_LJ_ene);
    controller.Step_Print("nb14A_EE", TI_core.data.nb14A_EE_ene);
    controller.Step_Print("nb14B_EE", TI_core.data.nb14B_EE_ene);
    controller.Step_Print("bond_soft", TI_core.data.bond_soft_dH_dlambda);
    controller.Step_Print("LJ(sc.)", TI_core.data.lj_soft_dH_dlambda);
    controller.Step_Print("Coul(direct.)", TI_core.data.coul_direct_dH_dlambda);
    controller.Step_Print("LR_corr(sc.)", TI_core.data.lj_soft_long_range_correction);
    controller.Step_Print("PME(reci.)", TI_core.data.pme_reci_dH_dlambda);
    controller.Step_Print("PME(self.)", TI_core.data.pme_self_dH_dlambda);
    controller.Step_Print("PME(corr.)", TI_core.data.pme_corr_dH_dlambda);

    controller.Print_To_Screen_And_Mdout();
}

void Main_Volume_Update()
{
    lj_soft.Update_Volume(TI_core.box_length);
    pme.Update_Volume(TI_core.box_length);
    neighbor_list.Update_Volume(TI_core.box_length);
}

void Main_Iteration()
{
    TI_core.Read_Next_Frame();
    Main_Volume_Update();
    neighbor_list.Neighbor_List_Update(TI_core.crd, TI_core.nb.d_excluded_list_start, TI_core.nb.d_excluded_list, TI_core.nb.d_excluded_numbers, 1);
}

void Main_Clear()
{
    bondA.Clear();
    bondB.Clear();
    bond_soft.Clear();
    angleA.Clear();
    angleB.Clear();
    dihedralA.Clear();
    dihedralB.Clear();
    nb14A.Clear();
    nb14B.Clear();
    neighbor_list.Clear();
    lj_soft.Clear();
    pme.Clear();
    //kinetic.Clear();
    TI_core.Clear();
    controller.Clear();

    fcloseall();
}
