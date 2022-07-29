#include "main_fep.cuh"

CONTROLLER controller;
FEP_CORE fep_core;
BOND bond;
BOND_SOFT bond_soft;
ANGLE angle;
DIHEDRAL dihedral;
NON_BOND_14 nb14;
NEIGHBOR_LIST neighbor_list;
LJ_SOFT_CORE lj_soft;
Particle_Mesh_Ewald pme;

int main(int argc, char *argv[])
{
    Main_Initial(argc, argv);
    for (fep_core.input.current_frame = 1; fep_core.input.current_frame <= fep_core.input.frame_numbers; ++fep_core.input.current_frame)
    {
        Main_Calculation();
        Main_Print();
        if (fep_core.input.current_frame != fep_core.input.frame_numbers)
        {
            Main_Iteration();
        }
    }
    fep_core.Print_Pure_Ene_To_Result_File();
    Main_Clear();
    return 0;
}

void Main_Initial(int argc, char *argv[])
{
    controller.Initial(argc, argv);
    fep_core.Initial(&controller);
    
    
    nb14.Initial(&controller, NULL, NULL, NULL);
    bond.Initial(&controller);
    bond_soft.Initial(&controller);
    angle.Initial(&controller);
    dihedral.Initial(&controller);

    lj_soft.Initial(&controller, fep_core.nb.cutoff, fep_core.box_length);

    if (fep_core.charge_pertubated)
        pme.Initial(&controller, fep_core.atom_numbers, fep_core.box_length, fep_core.nb.cutoff);

    if (fep_core.charge_pertubated || lj_soft.is_initialized)
    {
        //when we need to calculate direct energy
        neighbor_list.Initial(&controller, fep_core.atom_numbers, fep_core.box_length, fep_core.nb.cutoff, fep_core.nb.skin);
        neighbor_list.Neighbor_List_Update(fep_core.crd, fep_core.nb.d_excluded_list_start, fep_core.nb.d_excluded_list, fep_core.nb.d_excluded_numbers, 1);
    }

    controller.Input_Check();
    controller.Print_First_Line_To_Mdout();
}

void Main_Calculation()
{
    fep_core.data.current_frame_ene = 0.0;
    if (bond.is_initialized)
        fep_core.data.partition.bond_ene = bond.Get_Energy(fep_core.uint_crd, fep_core.pbc.uint_dr_to_dr_cof);
    else
        fep_core.data.partition.bond_ene = 0.0;
    
    if (angle.is_initialized)
        fep_core.data.partition.angle_ene = angle.Get_Energy(fep_core.uint_crd, fep_core.pbc.uint_dr_to_dr_cof);
    else
        fep_core.data.partition.angle_ene = 0.0;
    
    if (dihedral.is_initialized)
        fep_core.data.partition.dihedral_ene = dihedral.Get_Energy(fep_core.uint_crd, fep_core.pbc.uint_dr_to_dr_cof);
    else
        fep_core.data.partition.dihedral_ene = 0.0;
    
    if (nb14.is_initialized    )
    {
        fep_core.data.partition.nb14_EE_ene = nb14.Get_14_CF_Energy(fep_core.uint_crd, fep_core.d_charge, fep_core.pbc.uint_dr_to_dr_cof);
        fep_core.data.partition.nb14_LJ_ene = nb14.Get_14_LJ_Energy(fep_core.uint_crd, fep_core.pbc.uint_dr_to_dr_cof);
    }
    else
    {
        fep_core.data.partition.nb14_EE_ene = fep_core.data.partition.nb14_LJ_ene = 0.0;
    }

    if (bond_soft.is_initialized)
        fep_core.data.partition.bond_soft_ene = bond_soft.Get_Energy(fep_core.uint_crd, fep_core.pbc.uint_dr_to_dr_cof);
    else
        fep_core.data.partition.bond_soft_ene = 0.0;


    if (lj_soft.is_initialized)
    {
        fep_core.data.lj_soft_ene = lj_soft.Get_Energy_With_Coulomb_Direct(fep_core.uint_crd,neighbor_list.d_nl, fep_core.d_charge);
        fep_core.data.partition.vdw_intersys_ene = lj_soft.h_LJ_energy_sum_intersys;
        fep_core.data.partition.vdw_intrasys_ene = lj_soft.h_LJ_energy_sum_intrasys;
        fep_core.data.partition.coul_direct_intersys_ene = lj_soft.h_direct_ene_sum_intersys;
        fep_core.data.partition.coul_direct_intrasys_ene = lj_soft.h_direct_ene_sum_intrasys;
        fep_core.data.partition.vdw_long_range_correction = lj_soft.Long_Range_Correction();
    }
    else
    {
        fep_core.data.lj_soft_ene = 0.0;
        fep_core.data.partition.vdw_intersys_ene = 0.0;
        fep_core.data.partition.vdw_intrasys_ene = 0.0;
        fep_core.data.partition.coul_direct_intersys_ene = 0.0;
        fep_core.data.partition.coul_direct_intrasys_ene = 0.0;
    }

    
    if (fep_core.charge_pertubated)
    {
        fep_core.data.partition.coul_long_range = pme.Get_Energy(fep_core.uint_crd, fep_core.d_charge, neighbor_list.d_nl, fep_core.pbc.uint_dr_to_dr_cof,
            fep_core.nb.d_excluded_list_start, fep_core.nb.d_excluded_list, fep_core.nb.d_excluded_numbers, pme.CORRECTION);
    
        fep_core.data.partition.coul_long_range += pme.Get_Energy(fep_core.uint_crd, fep_core.d_charge, neighbor_list.d_nl, fep_core.pbc.uint_dr_to_dr_cof,
            fep_core.nb.d_excluded_list_start, fep_core.nb.d_excluded_list, fep_core.nb.d_excluded_numbers, pme.SELF);
    
        fep_core.data.partition.coul_long_range += pme.Get_Energy(fep_core.uint_crd, fep_core.d_charge, neighbor_list.d_nl, fep_core.pbc.uint_dr_to_dr_cof,
            fep_core.nb.d_excluded_list_start, fep_core.nb.d_excluded_list, fep_core.nb.d_excluded_numbers, pme.RECIPROCAL);
        
        if (!lj_soft.is_initialized)
            fep_core.Seperate_Direct_Atom_Energy(neighbor_list.d_nl, pme.beta);
    }
    else
    {
        fep_core.data.partition.coul_long_range = 0.0;
    }

    fep_core.data.partition.pV = fep_core.data.pressure * fep_core.box_length.x * fep_core.box_length.y * fep_core.box_length.z;

    fep_core.data.Sum_One_Frame(fep_core.input.current_frame);
}

void Main_Print()
{
    controller.Step_Print("frame", fep_core.input.current_frame);
    controller.Step_Print("ene", fep_core.data.current_frame_ene);
    controller.Step_Print("bond", fep_core.data.partition.bond_ene);
    controller.Step_Print("angle", fep_core.data.partition.angle_ene);
    controller.Step_Print("dihedral", fep_core.data.partition.dihedral_ene);
    controller.Step_Print("nb14_LJ", fep_core.data.partition.nb14_LJ_ene);
    controller.Step_Print("nb14_EE", fep_core.data.partition.nb14_EE_ene);
    controller.Step_Print("bond_soft", fep_core.data.partition.bond_soft_ene);
    controller.Step_Print("LJ(sc.)", fep_core.data.lj_soft_ene);
    controller.Step_Print("Coul(direct.)", fep_core.data.partition.coul_direct_intrasys_ene + fep_core.data.partition.coul_direct_intersys_ene);
    controller.Step_Print("LR_corr(sc.)", fep_core.data.partition.vdw_long_range_correction);
    controller.Step_Print("PME(reci.)", pme.reciprocal_ene);
    controller.Step_Print("PME(self.)", pme.self_ene);
    controller.Step_Print("PME(corr.)", pme.correction_ene);
    controller.Step_Print("Coul(all.)", fep_core.data.partition.coul_long_range + fep_core.data.partition.coul_direct_intersys_ene + fep_core.data.partition.coul_direct_intrasys_ene);
    controller.Step_Print("pV", fep_core.data.partition.pV);

    controller.Print_To_Screen_And_Mdout();
}

void Main_Volume_Update()
{
    lj_soft.Update_Volume(fep_core.box_length);
    pme.Update_Volume(fep_core.box_length);
    neighbor_list.Update_Volume(fep_core.box_length);
}

void Main_Iteration()
{
    fep_core.Read_Next_Frame();
    Main_Volume_Update();
    neighbor_list.Neighbor_List_Update(fep_core.crd, fep_core.nb.d_excluded_list_start, fep_core.nb.d_excluded_list, fep_core.nb.d_excluded_numbers, 1);
}

void Main_Clear()
{
    bond.Clear();
    bond_soft.Clear();
    angle.Clear();
    dihedral.Clear();
    nb14.Clear();
    neighbor_list.Clear();
    lj_soft.Clear();
    pme.Clear();
    fep_core.Clear();
    controller.Clear();
    fcloseall();
}