#include "main.cuh"

CONTROLLER controller;
MD_INFORMATION md_info;
MIDDLE_Langevin_INFORMATION middle_langevin;
Langevin_MD_INFORMATION langevin;
ANDERSEN_THERMOSTAT_INFORMATION ad_thermo;
BERENDSEN_THERMOSTAT_INFORMATION bd_thermo;
NOSE_HOOVER_CHAIN_INFORMATION nhc;
BOND bond;
ANGLE angle;
UREY_BRADLEY urey_bradley;
DIHEDRAL dihedral;
IMPROPER_DIHEDRAL improper;
NON_BOND_14 nb14;
CMAP cmap;
NEIGHBOR_LIST neighbor_list;
LENNARD_JONES_INFORMATION lj;
Particle_Mesh_Ewald pme;
RESTRAIN_INFORMATION restrain;
CONSTRAIN constrain;
SETTLE settle;
SIMPLE_CONSTRAIN simple_constrain;
SHAKE shake;
VIRTUAL_INFORMATION vatom;
CoordinateMolecularMap mol_map;
MC_BAROSTAT_INFORMATION mc_baro;
BERENDSEN_BAROSTAT_INFORMATION bd_baro;
ANDERSEN_BAROSTAT_INFORMATION ad_baro;
LJ_SOFT_CORE lj_soft;
BOND_SOFT bond_soft;

int main(int argc, char *argv[])
{
    Main_Initial(argc, argv);

    for (md_info.sys.steps = 1; md_info.sys.steps <= md_info.sys.step_limit; md_info.sys.steps++)
    {
        Main_Calculate_Force();
        Main_Iteration();
        Main_Print();
    }

    Main_Clear();
    return 0;
}

void Main_Initial(int argc, char *argv[])
{
    controller.Initial(argc, argv);
    md_info.Initial(&controller);
    controller.Command_Exist("end_pause");

    if  (md_info.mode >= md_info.NVT && controller.Command_Choice("thermostat", "middle_langevin"))
    {
        middle_langevin.Initial(&controller, md_info.atom_numbers, md_info.sys.target_temperature, md_info.h_mass);
    }
    if (md_info.mode >= md_info.NVT && controller.Command_Choice("thermostat", "langevin"))
    {
        langevin.Initial(&controller, md_info.atom_numbers, md_info.sys.target_temperature, md_info.h_mass);
    }
    if (md_info.mode >= md_info.NVT && controller.Command_Choice("thermostat", "berendsen_thermostat"))
    {
        bd_thermo.Initial(&controller, md_info.sys.target_temperature);
    }
    if (md_info.mode >= md_info.NVT && controller.Command_Choice("thermostat", "andersen_thermostat"))
    {
        ad_thermo.Initial(&controller, md_info.sys.target_temperature, md_info.atom_numbers, md_info.h_mass);
    }
    if (md_info.mode >= md_info.NVT && controller.Command_Choice("thermostat", "nose_hoover_chain"))
    {
        nhc.Initial(&controller, md_info.sys.target_temperature);
    }

    neighbor_list.Initial(&controller, md_info.atom_numbers, md_info.sys.box_length, md_info.nb.cutoff, md_info.nb.skin);
    neighbor_list.Neighbor_List_Update(md_info.crd, md_info.nb.d_excluded_list_start, md_info.nb.d_excluded_list, md_info.nb.d_excluded_numbers, neighbor_list.FORCED_UPDATE);
    lj.Initial(&controller, md_info.nb.cutoff, md_info.sys.box_length);
    lj_soft.Initial(&controller, md_info.nb.cutoff, md_info.sys.box_length);
    pme.Initial(&controller, md_info.atom_numbers, md_info.sys.box_length, md_info.nb.cutoff);
    
    nb14.Initial(&controller, lj.h_LJ_A, lj.h_LJ_B, lj.h_atom_LJ_type);
    bond.Initial(&controller);
    bond_soft.Initial(&controller);
    angle.Initial(&controller);
    urey_bradley.Initial(&controller);
    dihedral.Initial(&controller);
    improper.Initial(&controller);
    cmap.Initial(&controller);

    restrain.Initial(&controller, md_info.atom_numbers, md_info.crd);

    if (controller.Command_Exist("constrain_mode"))
    {    
        constrain.Add_HBond_To_Constrain_Pair(&controller, bond.bond_numbers, bond.h_atom_a, bond.h_atom_b, bond.h_r0, md_info.h_mass);
        constrain.Add_HAngle_To_Constrain_Pair(&controller, angle.angle_numbers, angle.h_atom_a, angle.h_atom_b, angle.h_atom_c, angle.h_angle_theta0, md_info.h_mass);
        if (middle_langevin.is_initialized)
            constrain.Initial_Constrain(&controller, md_info.atom_numbers, md_info.dt, md_info.sys.box_length, middle_langevin.exp_gamma, 0, md_info.h_mass, &md_info.sys.freedom);
        else
            constrain.Initial_Constrain(&controller, md_info.atom_numbers, md_info.dt, md_info.sys.box_length, 1.0, md_info.mode == md_info.MINIMIZATION, md_info.h_mass, &md_info.sys.freedom);
        if (!(controller.Command_Exist("settle_disable") && atoi(controller.Command("settle_disable")) != 0))
        {
            settle.Initial(&controller, &constrain, md_info.h_mass);
        }
        if (controller.Command_Choice("constrain_mode", "simple_constrain"))
        {
            simple_constrain.Initial_Simple_Constrain(&controller, &constrain);
        }
        if (controller.Command_Choice("constrain_mode", "shake"))
        {
            shake.Initial_Simple_Constrain(&controller, &constrain);
        }
    }


    if (md_info.mode == md_info.NPT && controller.Command_Choice("barostat", "monte_carlo_barostat"))
    {
        mc_baro.Initial(&controller, md_info.atom_numbers, md_info.sys.target_pressure, md_info.sys.box_length, md_info.res.is_initialized);
    }
    if (md_info.mode == md_info.NPT && controller.Command_Choice("barostat", "berendsen_barostat"))
    {
        bd_baro.Initial(&controller, md_info.sys.target_pressure, md_info.sys.box_length);
    }
    if (md_info.mode == md_info.NPT && controller.Command_Choice("barostat", "andersen_barostat"))
    {
        ad_baro.Initial(&controller, md_info.sys.target_pressure, md_info.sys.box_length);
    }
    vatom.Initial(&controller, md_info.atom_numbers, &md_info.sys.freedom);
    mol_map.Initial(md_info.atom_numbers, md_info.sys.box_length, md_info.crd,
        md_info.nb.excluded_atom_numbers, md_info.nb.h_excluded_numbers, md_info.nb.h_excluded_list_start, md_info.nb.h_excluded_list);

    controller.Input_Check();
    controller.Print_First_Line_To_Mdout();
    controller.core_time.Start();
}

void Main_Clear()
{
    controller.core_time.Stop();
    controller.printf("Core Run Wall Time: %f second(s)\n", controller.core_time.time);
    if (md_info.mode != md_info.MINIMIZATION)
    {
        controller.simulation_speed = md_info.sys.steps * md_info.dt / CONSTANT_TIME_CONVERTION / controller.core_time.time * 86.4;
        controller.printf("Core Run Speed: %f ns/day\n", controller.simulation_speed);
    }
    else
    {
        controller.simulation_speed = md_info.sys.steps / controller.core_time.time * 3600;
        controller.printf("Core Run Speed: %f steps/hour\n", controller.simulation_speed);
    }
    fcloseall();

    if (controller.Command_Exist("end_pause"))
    {
        if (atoi(controller.Command("end_pause")) == 1)
        {
            printf("End Pause\n");
            getchar();
        }
    }
}

void Main_Calculate_Force()
{

    if (md_info.mode == md_info.RERUN)
    {
        md_info.rerun.Iteration();
            Main_Box_Length_Change(md_info.rerun.box_length_change_factor);
    }
    md_info.MD_Information_Crd_To_Uint_Crd();
    md_info.MD_Reset_Atom_Energy_And_Virial_And_Force();
    if (md_info.sys.steps % md_info.output.write_mdout_interval == 0 || (md_info.mode == md_info.MINIMIZATION && md_info.min.dynamic_dt))
    {
        md_info.need_potential = 1;
    }
    mc_baro.Ask_For_Calculate_Potential(md_info.sys.steps, &md_info.need_potential);
    bd_baro.Ask_For_Calculate_Pressure(md_info.sys.steps, &md_info.need_pressure);
    if (ad_baro.is_initialized) md_info.need_pressure = 1;

    lj.LJ_PME_Direct_Force_With_Atom_Energy_And_Virial(md_info.atom_numbers, md_info.uint_crd, md_info.d_charge, md_info.frc,
        neighbor_list.d_nl, pme.beta, md_info.need_potential, md_info.d_atom_energy, md_info.need_pressure, md_info.d_atom_virial, pme.d_direct_atom_energy);
    lj.Long_Range_Correction(md_info.need_pressure, md_info.sys.d_virial,
        md_info.need_potential, md_info.sys.d_potential);

    lj_soft.LJ_Soft_Core_PME_Direct_Force_With_Atom_Energy_And_Virial(md_info.atom_numbers, md_info.uint_crd, md_info.d_charge, md_info.frc,
        neighbor_list.d_nl, pme.beta, md_info.need_potential, md_info.d_atom_energy, md_info.need_pressure, md_info.d_atom_virial, pme.d_direct_atom_energy);
    lj_soft.Long_Range_Correction(md_info.need_pressure, md_info.sys.d_virial,
        md_info.need_potential, md_info.sys.d_potential);
    
    pme.PME_Excluded_Force_With_Atom_Energy(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof, md_info.d_charge,
        md_info.nb.d_excluded_list_start, md_info.nb.d_excluded_list, md_info.nb.d_excluded_numbers, md_info.frc, pme.d_correction_atom_energy);

    pme.PME_Reciprocal_Force_With_Energy_And_Virial(md_info.uint_crd, md_info.d_charge, md_info.frc, md_info.need_pressure, md_info.need_potential, md_info.sys.d_virial, md_info.sys.d_potential);

    nb14.Non_Bond_14_LJ_CF_Force_With_Atom_Energy_And_Virial(md_info.uint_crd, md_info.d_charge, md_info.pbc.uint_dr_to_dr_cof, md_info.frc, md_info.d_atom_energy, md_info.d_atom_virial);

    bond.Bond_Force_With_Atom_Energy_And_Virial(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof, md_info.frc, md_info.d_atom_energy, md_info.d_atom_virial);
    bond_soft.Soft_Bond_Force_With_Atom_Energy_And_Virial(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof, md_info.frc, md_info.d_atom_energy, md_info.d_atom_virial);
    angle.Angle_Force_With_Atom_Energy(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof, md_info.frc, md_info.d_atom_energy);
    urey_bradley.Urey_Bradley_Force_With_Atom_Energy_And_Virial(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof, md_info.frc, md_info.d_atom_energy, md_info.d_atom_virial);
    dihedral.Dihedral_Force_With_Atom_Energy(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof, md_info.frc, md_info.d_atom_energy);
    improper.Dihedral_Force_With_Atom_Energy(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof, md_info.frc, md_info.d_atom_energy);
    cmap.CMAP_Force_with_Atom_Energy(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof, md_info.frc, md_info.d_atom_energy);

    restrain.Restraint(md_info.crd, md_info.sys.box_length, md_info.d_atom_energy, md_info.d_atom_virial, md_info.frc);

    vatom.Force_Redistribute(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof, md_info.frc);

    md_info.Calculate_Pressure_And_Potential_If_Needed();
}

void Main_Iteration()
{
    if (md_info.mode == md_info.RERUN)
    {
        neighbor_list.Neighbor_List_Update(md_info.crd, md_info.nb.d_excluded_list_start, md_info.nb.d_excluded_list, md_info.nb.d_excluded_numbers);
        return;
    }
    //\B8\C3\C0\A8\BA\C5\CA\F4\D3\DAmc\BF\D8ѹ\B2\BF\B7\D6
    if (mc_baro.is_initialized && md_info.sys.steps % mc_baro.update_interval == 0)
    {
        //\BE\C9\C4\DC\C1\BF
        mc_baro.energy_old = md_info.sys.h_potential;
        cudaMemcpy(mc_baro.frc_backup, md_info.frc, sizeof(VECTOR)*md_info.atom_numbers, cudaMemcpyDeviceToDevice);
        cudaMemcpy(mc_baro.crd_backup, md_info.crd, sizeof(VECTOR)*md_info.atom_numbers, cudaMemcpyDeviceToDevice);

        mc_baro.Volume_Change_Attempt(md_info.sys.box_length);


        //\B8ı\E4\D7\F8\B1\EA
        if (mc_baro.scale_coordinate_by_molecule)
        {
            mol_map.Calculate_No_Wrap_Crd(md_info.crd);
            md_info.mol.Molecule_Crd_Map(mol_map.nowrap_crd, mc_baro.crd_scale_factor);
            mol_map.Refresh_BoxMapTimes(md_info.crd);
        }
        else
        {
            mc_baro.Scale_Coordinate_Atomically(md_info.atom_numbers, md_info.crd);
        }

        
        //\B8ı\E4\CC\E5\BB\FD
        Main_Box_Length_Change(mc_baro.crd_scale_factor);

        //\D0\C2\C4\DC\C1\BF
        Main_Calculate_Force();
        mc_baro.energy_new = md_info.sys.h_potential;

        //\BC\C6\CB\E3\BD\D3\CAܸ\C5\C2\CA
        if (mc_baro.scale_coordinate_by_molecule)
            mc_baro.extra_term = md_info.sys.target_pressure * mc_baro.DeltaV - md_info.mol.molecule_numbers * CONSTANT_kB * md_info.sys.target_temperature * logf(mc_baro.VDevided);
        else
            mc_baro.extra_term = md_info.sys.target_pressure * mc_baro.DeltaV - md_info.atom_numbers * CONSTANT_kB * md_info.sys.target_temperature * logf(mc_baro.VDevided);

        if (mc_baro.couple_dimension != mc_baro.NO && mc_baro.couple_dimension != mc_baro.XYZ)
            mc_baro.extra_term -= mc_baro.surface_number * mc_baro.surface_tension * mc_baro.DeltaS * mc_baro.TENSION_UNIT_FACTOR;

        mc_baro.accept_possibility = mc_baro.energy_new - mc_baro.energy_old + mc_baro.extra_term;
        mc_baro.accept_possibility = expf(-mc_baro.accept_possibility / (CONSTANT_kB * md_info.sys.target_temperature));

        //\C5ж\CF\CAǷ\F1\BD\D3\CA\DC
        if (mc_baro.Check_MC_Barostat_Accept())
        {
            //\B1\BB\BEܾ\F8\C1˾ͻ\B9ԭ
            mc_baro.crd_scale_factor = 1.0 / mc_baro.crd_scale_factor;
            cudaMemcpy(md_info.crd, mc_baro.crd_backup, sizeof(VECTOR)*md_info.atom_numbers, cudaMemcpyDeviceToDevice);
            Main_Box_Length_Change(mc_baro.crd_scale_factor);
            neighbor_list.Neighbor_List_Update(md_info.crd, md_info.nb.d_excluded_list_start, md_info.nb.d_excluded_list, md_info.nb.d_excluded_numbers, neighbor_list.CONDITIONAL_UPDATE, neighbor_list.FORCED_CHECK);
            cudaMemcpy(md_info.frc, mc_baro.frc_backup, sizeof(VECTOR)*md_info.atom_numbers, cudaMemcpyDeviceToDevice);
        }
        //\BD\D3\CAܺ\F3\CC\E5\BB\FD\B1仯\B9\FD\B4\F3\BB\F2\B6\D4\CC\E5\BB\FD\B2\D9\D7\F7\B4\CE\CA\FD̫\B6ࣨ~1 ns\A3\A9\D2Ժ\F3\A3\AC\D6\D8\D0¶Բ\BF\B7\D6ģ\BF\E9\B3\F5ʼ\BB\AF
        if ((!mc_baro.reject && (mc_baro.newV > 1.331 * mc_baro.V0 || mc_baro.newV < 0.729 * mc_baro.V0)))
        {
            Main_Volume_Change_Largely();
            mc_baro.V0 = mc_baro.newV;
        }

        //\B6\D4\D7\EE\B4\F3\B1仯ֵ\BD\F8\D0е\FC\B4\FA
        mc_baro.Delta_Box_Length_Max_Update();
    }
    
    settle.Remember_Last_Coordinates(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof);
    simple_constrain.Remember_Last_Coordinates(md_info.crd, md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof);
    shake.Remember_Last_Coordinates(md_info.crd, md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof);

    if (md_info.mode == md_info.NVE)
    {
        md_info.nve.Leap_Frog();
    }
    else if (md_info.mode == md_info.MINIMIZATION)
    {
        md_info.min.Gradient_Descent();
    }
    else if (middle_langevin.is_initialized)
    {
        middle_langevin.MD_Iteration_Leap_Frog(md_info.frc, md_info.vel, md_info.acc, md_info.crd);
    }
    else if (langevin.is_initialized)
    {
        langevin.MD_Iteration_Leap_Frog(md_info.frc, md_info.crd, md_info.vel, md_info.acc);
    }
    else if (bd_thermo.is_initialized)
    {
        bd_thermo.Record_Temperature(md_info.sys.Get_Atom_Temperature(), md_info.sys.freedom);
        md_info.nve.Leap_Frog();
        bd_thermo.Scale_Velocity(md_info.atom_numbers, md_info.vel);
    }
    else if (ad_thermo.is_initialized)
    {
        if ((md_info.sys.steps - 1) % ad_thermo.update_interval == 0)
        {
            ad_thermo.MD_Iteration_Leap_Frog(md_info.atom_numbers, md_info.vel, md_info.crd, md_info.frc, md_info.acc, md_info.d_mass_inverse, md_info.dt);
            constrain.v_factor = FLT_MIN;
            constrain.x_factor = 0.5;
        }
        else
        {
            md_info.nve.Leap_Frog();
            constrain.v_factor = 1.0;
            constrain.x_factor = 1.0;
        }
    }
    else if (nhc.is_initialized)
    {
        nhc.MD_Iteration_Leap_Frog(md_info.atom_numbers, md_info.vel, md_info.crd, md_info.frc, md_info.acc, md_info.d_mass_inverse, md_info.dt, md_info.sys.Get_Total_Atom_Ek(), md_info.sys.freedom);
    }

    settle.Do_SETTLE(md_info.d_mass, md_info.crd, md_info.sys.box_length, md_info.vel, md_info.need_pressure, md_info.sys.d_pressure);
    simple_constrain.Constrain(md_info.crd, md_info.vel, md_info.d_mass_inverse, md_info.d_mass, md_info.sys.box_length, md_info.need_pressure, md_info.sys.d_pressure);
    shake.Constrain(md_info.crd, md_info.vel, md_info.d_mass_inverse, md_info.d_mass, md_info.sys.box_length, md_info.need_pressure, md_info.sys.d_pressure);
    

    if (bd_baro.is_initialized && md_info.sys.steps % bd_baro.update_interval == 0)
    {
        cudaMemcpy(&md_info.sys.h_pressure, md_info.sys.d_pressure, sizeof(float), cudaMemcpyDeviceToHost);
        float p_now = md_info.sys.h_pressure;
        bd_baro.crd_scale_factor = 1 - bd_baro.update_interval * bd_baro.compressibility * bd_baro.dt / bd_baro.taup / 3 * (md_info.sys.target_pressure - p_now);

        if (bd_baro.stochastic_term)
        {
            bd_baro.crd_scale_factor += sqrtf(2 * CONSTANT_kB * md_info.sys.target_temperature* bd_baro.compressibility / bd_baro.taup / bd_baro.newV)
                / 3 * bd_baro.n(bd_baro.e);
            Scale_List((float*)md_info.vel, 1.0f / bd_baro.crd_scale_factor, 3 * md_info.atom_numbers);
        }
        Scale_List((float*)md_info.crd, bd_baro.crd_scale_factor, 3 * md_info.atom_numbers);
        Main_Volume_Change(bd_baro.crd_scale_factor);
        bd_baro.newV = md_info.sys.Get_Volume();
        if (bd_baro.newV > 1.331 * bd_baro.V0 || bd_baro.newV < 0.729 * bd_baro.V0)
        {
            Main_Volume_Change_Largely();
            bd_baro.V0 = bd_baro.newV;
        }
    }

    if (ad_baro.is_initialized)
    {
        cudaMemcpy(&md_info.sys.h_pressure, md_info.sys.d_pressure, sizeof(float), cudaMemcpyDeviceToHost);
        ad_baro.dV_dt += (md_info.sys.h_pressure - md_info.sys.target_pressure) * ad_baro.h_mass_inverse;
        ad_baro.new_V = md_info.sys.Get_Volume() + ad_baro.dV_dt * md_info.dt;

        ad_baro.crd_scale_factor = cbrt(ad_baro.new_V / md_info.sys.Get_Volume());
        Scale_List((float*)md_info.crd, ad_baro.crd_scale_factor, 3 * md_info.atom_numbers);
        Scale_List((float*)md_info.vel, 1.0 / ad_baro.crd_scale_factor, 3 * md_info.atom_numbers);
        Main_Volume_Change(ad_baro.crd_scale_factor);

        if (ad_baro.new_V > 1.331 * ad_baro.V0 || ad_baro.new_V < 0.729 * ad_baro.V0)
        {
            Main_Volume_Change_Largely();
            ad_baro.V0 = ad_baro.new_V;
        }
    }

    md_info.MD_Information_Crd_To_Uint_Crd();
    vatom.Coordinate_Refresh(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof, md_info.crd);//ע\D2\E2\B8\FC\D0\C2uint crd
    neighbor_list.Neighbor_List_Update(md_info.crd, md_info.nb.d_excluded_list_start, md_info.nb.d_excluded_list, md_info.nb.d_excluded_numbers);
    mol_map.Refresh_BoxMapTimes(md_info.crd);
}

void Main_Print()
{
    if (md_info.sys.steps % md_info.output.write_mdout_interval == 0)
    {
        controller.Step_Print("step", md_info.sys.steps);
        controller.Step_Print("time", md_info.sys.Get_Current_Time());
        controller.Step_Print("temperature", md_info.res.Get_Residue_Temperature());
        controller.Step_Print("potential", md_info.sys.h_potential);
        controller.Step_Print("PME", pme.Get_Energy(md_info.uint_crd, md_info.d_charge, neighbor_list.d_nl, md_info.pbc.uint_dr_to_dr_cof,
            md_info.nb.d_excluded_list_start, md_info.nb.d_excluded_list, md_info.nb.d_excluded_numbers));
        controller.Step_Print("LJ", lj.Get_Energy(md_info.uint_crd, neighbor_list.d_nl));
        lj_soft.Get_Energy(md_info.uint_crd, neighbor_list.d_nl, pme.beta, md_info.d_charge, pme.d_direct_ene);
        controller.Step_Print("LJ(sc.)", lj_soft.h_LJ_energy_sum);
        controller.Step_Print("LR_corr(sc.)", lj_soft.long_range_correction);
        controller.Step_Print("nb14_LJ", nb14.Get_14_LJ_Energy(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof));
        controller.Step_Print("nb14_EE", nb14.Get_14_CF_Energy(md_info.uint_crd, md_info.d_charge, md_info.pbc.uint_dr_to_dr_cof));
        controller.Step_Print("bond", bond.Get_Energy(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof));
        controller.Step_Print("bond_soft", bond_soft.Get_Energy(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof));
        controller.Step_Print("angle", angle.Get_Energy(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof));
        controller.Step_Print("urey_bradley", urey_bradley.Get_Energy(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof));
        controller.Step_Print("restrain", restrain.Get_Energy(md_info.crd, md_info.sys.box_length));
        controller.Step_Print("dihedral", dihedral.Get_Energy(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof));
        controller.Step_Print("improper_dihedral", improper.Get_Energy(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof));
        controller.Step_Print("cmap", cmap.Get_Energy(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof));
        controller.Step_Print("density", md_info.sys.Get_Density());
        controller.Step_Print("pressure", md_info.sys.h_pressure * CONSTANT_PRES_CONVERTION);
        controller.Step_Print("dV/dt", ad_baro.dV_dt);

        controller.Print_To_Screen_And_Mdout();
    }
    if (md_info.output.write_trajectory_interval && md_info.sys.steps % md_info.output.write_trajectory_interval == 0)
    {
        if (md_info.output.is_molecule_map_output)
        {
            mol_map.Calculate_No_Wrap_Crd(md_info.crd);
            md_info.mol.Molecule_Crd_Map(mol_map.nowrap_crd);
            mol_map.Refresh_BoxMapTimes(md_info.crd);
        }
        md_info.output.Append_Crd_Traj_File();
        md_info.output.Append_Box_Traj_File();
        // 20210827\D3\C3\D3\DA\CA\E4\B3\F6\CBٶȺ\CD\C1\A6
        if (md_info.output.is_vel_traj)
        {
            md_info.output.Append_Vel_Traj_File();
        }
        if (md_info.output.is_frc_traj)
        {
            md_info.output.Append_Frc_Traj_File();
        }
        nhc.Save_Trajectory_File();
    }
    if (md_info.output.write_restart_file_interval && md_info.sys.steps % md_info.output.write_restart_file_interval == 0)
    {
        md_info.output.Export_Restart_File();
        nhc.Save_Restart_File();
    }
}

void Main_Volume_Change(double factor)
{
    md_info.Update_Volume(factor);
    neighbor_list.Update_Volume(md_info.sys.box_length);
    neighbor_list.Neighbor_List_Update(md_info.crd, md_info.nb.d_excluded_list_start, md_info.nb.d_excluded_list, md_info.nb.d_excluded_numbers, neighbor_list.CONDITIONAL_UPDATE, neighbor_list.FORCED_CHECK);
    lj.Update_Volume(md_info.sys.box_length);
    lj_soft.Update_Volume(md_info.sys.box_length);
    pme.Update_Volume(md_info.sys.box_length);
    constrain.Update_Volume(md_info.sys.box_length);
    mol_map.Update_Volume(md_info.sys.box_length);
}

void Main_Box_Length_Change(VECTOR factor)
{
    md_info.Update_Box_Length(factor);
    neighbor_list.Update_Volume(md_info.sys.box_length);
    neighbor_list.Neighbor_List_Update(md_info.crd, md_info.nb.d_excluded_list_start, md_info.nb.d_excluded_list, md_info.nb.d_excluded_numbers, neighbor_list.CONDITIONAL_UPDATE, neighbor_list.FORCED_CHECK);
    lj.Update_Volume(md_info.sys.box_length);
    lj_soft.Update_Volume(md_info.sys.box_length);
    pme.Update_Box_Length(md_info.sys.box_length);
    constrain.Update_Volume(md_info.sys.box_length);
    mol_map.Update_Volume(md_info.sys.box_length);
}

void Main_Volume_Change_Largely()
{
    controller.printf("Some modules are based on the meshing methods, and it is more precise to re-initialize these modules now for a long time or a large volume change.\n");
    neighbor_list.Clear();
    pme.Clear();
    neighbor_list.Initial(&controller, md_info.atom_numbers, md_info.sys.box_length, md_info.nb.cutoff, md_info.nb.skin);
    neighbor_list.Neighbor_List_Update(md_info.crd, md_info.nb.d_excluded_list_start, md_info.nb.d_excluded_list, md_info.nb.d_excluded_numbers, 1);
    pme.Initial(&controller, md_info.atom_numbers, md_info.sys.box_length ,md_info.nb.cutoff );
    controller.printf("---------------------------------------------------------------------------------------\n"); 
}

