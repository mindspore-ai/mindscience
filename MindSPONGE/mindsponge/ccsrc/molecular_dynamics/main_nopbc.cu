#include "main_nopbc.cuh"

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

RESTRAIN_INFORMATION restrain;

CONSTRAIN constrain;
SETTLE settle;
SIMPLE_CONSTRAIN simple_constrain;
SHAKE shake;

VIRTUAL_INFORMATION vatom;


CoordinateMolecularMap mol_map;

LENNARD_JONES_NO_PBC_INFORMATION LJ_NOPBC;
COULOMB_FORCE_NO_PBC_INFORMATION CF_NOPBC;
GENERALIZED_BORN_INFORMATION gb;

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

    LJ_NOPBC.Initial(&controller, md_info.nb.cutoff);
    CF_NOPBC.Initial(&controller, md_info.atom_numbers, md_info.nb.cutoff);
    if (controller.Command_Exist("gb", "in_file"))
    {
        gb.Initial(&controller, md_info.nb.cutoff);
    }

    nb14.Initial(&controller, LJ_NOPBC.h_LJ_A, LJ_NOPBC.h_LJ_B, LJ_NOPBC.h_atom_LJ_type);
    bond.Initial(&controller);
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
    }
    md_info.MD_Information_Crd_To_Uint_Crd();
    md_info.MD_Reset_Atom_Energy_And_Virial_And_Force();
    mol_map.Calculate_No_Wrap_Crd(md_info.crd);


    if (md_info.sys.steps % md_info.output.write_mdout_interval == 0 || (md_info.mode == md_info.MINIMIZATION && md_info.min.dynamic_dt))
    {
        md_info.need_potential = 1;
    }

    LJ_NOPBC.LJ_Force_With_Atom_Energy(md_info.atom_numbers, mol_map.nowrap_crd, md_info.frc, md_info.need_potential, md_info.d_atom_energy, md_info.nb.d_excluded_list_start, md_info.nb.d_excluded_list, md_info.nb.d_excluded_numbers);
    CF_NOPBC.Coulomb_Force_With_Atom_Energy(md_info.atom_numbers, mol_map.nowrap_crd, md_info.d_charge, md_info.frc, md_info.need_potential, md_info.d_atom_energy, md_info.nb.d_excluded_list_start, md_info.nb.d_excluded_list, md_info.nb.d_excluded_numbers);
    gb.Get_Effective_Born_Radius(md_info.crd);
    gb.GB_Force_With_Atom_Energy(md_info.atom_numbers, mol_map.nowrap_crd, md_info.d_charge, md_info.frc, md_info.d_atom_energy);

    nb14.Non_Bond_14_LJ_CF_Force_With_Atom_Energy_And_Virial(md_info.uint_crd, md_info.d_charge, md_info.pbc.uint_dr_to_dr_cof, md_info.frc, md_info.d_atom_energy, md_info.d_atom_virial);

    bond.Bond_Force_With_Atom_Energy_And_Virial(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof, md_info.frc, md_info.d_atom_energy, md_info.d_atom_virial);
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
        return;
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
    

    md_info.MD_Information_Crd_To_Uint_Crd();
    vatom.Coordinate_Refresh(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof, md_info.crd);//ע\D2\E2\B8\FC\D0\C2uint crd
    mol_map.Refresh_BoxMapTimes(md_info.crd);
}

void Main_Print()
{
    if (md_info.sys.steps % md_info.output.write_mdout_interval == 0)
    {
        controller.Step_Print("step", md_info.sys.steps);
        controller.Step_Print("time", md_info.sys.Get_Current_Time());
        controller.Step_Print("temperature", md_info.sys.Get_Atom_Temperature());
        controller.Step_Print("potential", md_info.sys.h_potential);
        controller.Step_Print("Coulomb", CF_NOPBC.Get_Energy(md_info.crd, md_info.d_charge, md_info.nb.d_excluded_list_start, md_info.nb.d_excluded_list, md_info.nb.d_excluded_numbers));
        controller.Step_Print("gb", gb.Get_Energy(md_info.crd, md_info.d_charge));
        controller.Step_Print("LJ", LJ_NOPBC.Get_Energy(md_info.crd, md_info.nb.d_excluded_list_start, md_info.nb.d_excluded_list, md_info.nb.d_excluded_numbers));

        controller.Step_Print("nb14_LJ", nb14.Get_14_LJ_Energy(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof));
        controller.Step_Print("nb14_EE", nb14.Get_14_CF_Energy(md_info.uint_crd, md_info.d_charge, md_info.pbc.uint_dr_to_dr_cof));
        controller.Step_Print("bond", bond.Get_Energy(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof));
        controller.Step_Print("angle", angle.Get_Energy(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof));
        controller.Step_Print("urey_bradley", urey_bradley.Get_Energy(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof));
        controller.Step_Print("restrain", restrain.Get_Energy(md_info.crd, md_info.sys.box_length));
        controller.Step_Print("dihedral", dihedral.Get_Energy(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof));
        controller.Step_Print("improper_dihedral", improper.Get_Energy(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof));
        controller.Step_Print("cmap", cmap.Get_Energy(md_info.uint_crd, md_info.pbc.uint_dr_to_dr_cof));

        controller.Print_To_Screen_And_Mdout();
    }
    if (md_info.output.write_trajectory_interval && md_info.sys.steps % md_info.output.write_trajectory_interval == 0)
    {
        mol_map.Calculate_No_Wrap_Crd(md_info.crd);
        cudaMemcpy(md_info.coordinate, mol_map.nowrap_crd, sizeof(VECTOR)*md_info.atom_numbers, cudaMemcpyDeviceToHost);
        md_info.output.current_crd_synchronized_step = md_info.sys.steps;
        md_info.output.Append_Crd_Traj_File();
        md_info.output.Append_Box_Traj_File();
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

