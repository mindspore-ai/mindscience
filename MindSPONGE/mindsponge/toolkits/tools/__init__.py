"""
This **module** implements the terminal commands
"""
import os
import sys
import unittest
import multiprocessing as mpc
from ..helper import source, GlobalSetting, Xopen, Xprint
from ..mdrun import run


class TestMyPackage(unittest.TestCase):
    """
    This **class** does the unit tests for Xponge

    :param methodName: the method name of the unit test
    :param args: arguments from argparse
    :return: None
    """
    def __init__(self, methodName='runTest', args=None):
        super().__init__(methodName)
        self.args = args

    @classmethod
    def get_test_suite(cls, name, args):
        suite = unittest.TestSuite()
        test_name = "test_" + name
        suite.addTest(cls(test_name, args))
        return suite

    def test_base(self):
        """
        This **function** does the basic test for Xponge

        :param args: arguments from argparse
        :return: None
        """
        args = self.args
        source("..")
        source("..forcefield.amber.ff14sb")
        source("..forcefield.amber.tip3p")
        source("__main__")

        t = ACE + ALA * 10 + NME

        for i in range(1, len(t.residues) - 1):
            head = t.residues[i - 1]
            res = t.residues[i]
            tail = t.residues[i + 1]
            Impose_Dihedral(t, head.C, res.N, res.CA, res.C, -3.1415926 / 3)
            Impose_Dihedral(t, res.N, res.CA, res.C, tail.N, -3.1415926 / 3)

        Save_Mol2(t, f"{args.o}.mol2")
        c = int(round(t.charge))
        Add_Solvent_Box(t, WAT, 10)
        Solvent_Replace(t, lambda res: res.type.name == "WAT", {CL: 30 + c, K: 30})
        t.residues.sort(key=lambda residue: {"CL": 2, "K": 1, "WAT": 3}.get(residue.type.name, 0))
        Save_PDB(t, f"{args.o}.pdb")
        Save_SPONGE_Input(t, f"{args.o}")

        f = Xopen(f"{args.o}_LJ.txt", "r")
        atom_number, lj_number = [int(i) for i in f.readline().split()]
        self.assertEqual(atom_number, 4357, "LJ_in_file wrong")
        self.assertEqual(lj_number, 11, "LJ_in_file wrong")
        f.close()

        f = Xopen(f"{args.o}_dihedral.txt", "r")
        dihedral_number, = [int(i) for i in f.readline().split()]
        self.assertEqual(dihedral_number, 303, "dihedral_in_file wrong")
        f.close()

        f = Xopen(f"{args.o}_exclude.txt", "r")
        atom_number, exclude_number = [int(i) for i in f.readline().split()]
        self.assertEqual(atom_number, 4357, "exclude_in_file wrong")
        self.assertEqual(exclude_number, 4760, "exclude_in_file wrong")
        f.close()

    def test_charmm27(self):
        """
        This **function** does the CHARMM27 test for Xponge

        :param args: arguments from argparse
        :return: None
        """
        args = self.args
        source("..")
        AtomType.clear_type()
        ResidueType.clear_type()
        for frc in GlobalSetting.BondedForces:
            frc.clear_type()
        source("..forcefield.charmm27.protein")
        source("__main__")

        t = ACE + ALA * 10 + NME
        Save_SPONGE_Input(t, f"{args.o}")

    def test_assign(self):
        """
        This **function** does the assignment test for Xponge

        :param args: arguments from argparse
        :return: None
        """
        args = self.args
        source("..")
        source("..forcefield.amber.gaff")
        source("__main__")

        t = assign.Assign()
        t.add_atom("O", 0, 0, 0)
        t.add_atom("H", 1, 0, 0)
        t.add_atom("H", 0, 1, 0)
        t.add_bond(0, 1, 1)
        t.add_bond(0, 2, 1)
        t.determine_ring_and_bond_type()
        t.determine_atom_type("gaff")
        equal_atoms = t.Determine_Equal_Atoms()
        t.calculate_charge("resp", opt=True, extra_equivalence=equal_atoms)
        Save_PDB(t, f"{args.o}.pdb")
        Save_Mol2(t, f"{args.o}.mol2")
        wat = t.to_residuetype("WAT")
        Save_PDB(wat, f"{args.o}_Residue.pdb")
        Save_Mol2(wat, f"{args.o}_Residue.mol2")
        Save_SPONGE_Input(wat, f"{args.o}")

        t = load_mol2(f"{args.o}_Residue.mol2")
        wat_ = t.residues[0]
        diff = abs(wat_.O.charge + 0.8)
        self.assertLess(diff, 0.02)
        diff = abs(wat_.H.charge - 0.4)
        self.assertLess(diff, 0.02)
        diff = abs(wat_.H.charge - wat_.H1.charge)
        self.assertLess(diff, 0.01)

    def test_lattice(self):
        """
        This **function** does the assignment test for the lattice functions

        :param args: arguments from argparse
        :return: None
        """
        args = self.args
        source("..")
        source("..forcefield.amber.tip3p")
        source("__main__")
        box = BlockRegion(0, 0, 0, 60, 60, 60)
        region_1 = BlockRegion(0, 0, 20, 20, 20, 40)
        region_2 = BlockRegion(0, 0, 40, 20, 20, 60)
        region_3 = BlockRegion(0, 0, 0, 20, 20, 20)
        region_4 = SphereRegion(20, 10, 30, 10)
        region_5 = BlockRegion(0, 0, 0, 20, 20, 40, side="out")
        region_2or3 = UnionRegion(region_2, region_3)
        region_4and5 = IntersectRegion(region_4, region_5)
        region_6 = FrustumRegion(10, 40, 0, 15, 10, 40, 60, 1)
        region_7 = PrismRegion(30, 30, 0, 20, 0, 0, 0, 20, 0, 10, 10, 20)
        t = Lattice("bcc", basis_molecule=CL, scale=4)
        t2 = Lattice("fcc", basis_molecule=K, scale=3)
        t3 = Lattice("sc", basis_molecule=NA, scale=3)
        t4 = Lattice("hcp", basis_molecule=MG2, scale=4)
        t5 = Lattice("diamond", basis_molecule=AL3, scale=5)
        mol = t.Create(box, region_1)
        mol = t2.create(box, region_2or3, mol)
        mol = t3.create(box, region_4and5, mol)
        mol = t4.create(box, region_6, mol)
        mol = t5.create(box, region_7, mol)
        Save_PDB(mol, f"{args.o}.pdb")


def _one_test(ccon, name, args):
    """

    :param ccon:
    :param name:
    :param args:
    :return:
    """
    devnull = Xopen(os.devnull, 'w')
    sys.stdout = devnull
    sys.stderr = devnull
    runner = unittest.TextTestRunner(stream=devnull)
    t = runner.run(TestMyPackage.get_test_suite(name, args))
    ccon.send([t.errors, t.failures])
    devnull.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__


def test(args):
    """
    This **function** does the tests for Xponge

    :param args: arguments from argparse
    :return: None
    """
    GlobalSetting.verbose = args.verbose
    if not args.do:
        args.do = [["base"]]
    args.do = args.do[0]
    if "all" in args.do:
        args.do = ["base", "charmm27", "assign", "lattice"]
    fcon, ccon = mpc.Pipe()
    errors = []
    failures = []
    Xprint("Test(s): " + " ".join(args.do), verbose=-1)
    Xprint("======================================================", verbose=-1)
    for name in args.do:
        p = mpc.Process(target=_one_test, args=(ccon, name, args))
        p.start()
        p.join()
        new_errors, new_failures = fcon.recv()
        errors.extend(new_errors)
        failures.extend(new_failures)
        this_correct = f"{len(new_errors)} error(s) and {len(new_failures)} failure(s)"
        for ei, error in enumerate(new_errors):
            Xprint(f"Error {ei+1} for {name}", verbose=1)
            Xprint(f"{error[1]}", verbose=1)
        for ei, error in enumerate(new_failures):
            Xprint(f"Failure {ei+1} for {name}", verbose=1)
            Xprint(f"{error[1]}", verbose=1)
        Xprint(f"{name}: {this_correct}", verbose=-1)
    Xprint("======================================================", verbose=-1)
    if not errors and not failures:
        Xprint("No error or failure", verbose=-1)
    else:
        Xprint(f"{len(errors)} error(s) and {len(failures)} failure(s) found", verbose=-1)
        sys.exit(1)


def converter(args):
    """
    This **function** converts the format of coordinate file

    :param args: arguments from argparse
    :return: None
    """
    from ..analysis import md_analysis as xmda
    import MDAnalysis as mda

    if args.c:
        if args.cf == "guess":
            u = mda.Universe(args.p, args.c)
        elif args.cf == "sponge_crd":
            u = mda.Universe(args.p, args.c, format=xmda.SpongeCoordinateReader)
        elif args.cf == "sponge_traj":
            dirname, basename = os.path.split(args.c)
            if basename == "mdcrd.dat":
                box = "mdbox.txt"
            else:
                box = basename.replace(".dat", ".box")
            box = os.path.join(dirname, box)
            if box and os.path.exists(box):
                u = mda.Universe(args.p, args.c, box=box, format=xmda.SpongeTrajectoryReader)
            else:
                u = mda.Universe(args.p, args.c, format=xmda.SpongeTrajectoryReader)
    else:
        u = mda.Universe(args.p)

    if args.of == "sponge_crd":
        with xmda.SpongeCoordinateWriter(args.o) as w:
            w.write(u)
    elif args.of == "sponge_traj":
        with xmda.SpongeTrajectoryWriter(args.o) as w:
            for _ in u.trajectory:
                w.write(u)
    else:
        with mda.Writer(args.o, n_atoms=len(u.coord.positions)) as w:
            for _ in u.trajectory:
                w.write(u)


def maskgen(args):
    """
    This **function** uses VMD to generate mask

    :param args: arguments from argparse
    :return: None
    """
    s = input("Please Enter Your Selection Mask:\n")

    p = args.p.split(os.path.sep)
    p = "/".join(p)

    c = ""
    if args.c:
        c = args.c.split(os.path.sep)
        c = "/".join(c)
        c = "mol addfile " + c

    temp_write = """set f [open "{0}" "w"]
mol new {1}
{2}
atomselect top "{3}"
puts $f [atomselect0 list]
close $f
quit
""".format(args.o, p, c, s)

    temp = Xopen("maskgen_temp_tcl_file", "w")
    temp.write(temp_write)
    temp.close()

    os.system("{0} -dispdev none -e maskgen_temp_tcl_file".format(args.vmd))
    os.remove("maskgen_temp_tcl_file")


def exgen(args):
    """
    This **function** reads the SPONGE input files for bonded interactions and generate a exclude file

    :param args: arguments from argparse
    :return: None
    """
    partners = [set([]) for i in range(args.n)]

    def exclude_2_atoms(words):
        i, j = int(words[0]), int(words[1])
        partners[i].add(j)
        partners[j].add(i)

    def exclude_3_atoms(words):
        i, k = int(words[0]), int(words[2])
        partners[i].add(k)
        partners[k].add(i)

    def exclude_4_atoms(words):
        i, l = int(words[0]), int(words[3])
        partners[i].add(l)
        partners[l].add(i)

    for bond in args.bond:
        with open(bond) as f:
            f.readline()
            for line in f:
                words = line.split()
                exclude_2_atoms(words)

    for angle in args.angle:
        with open(angle) as f:
            f.readline()
            for line in f:
                words = line.split()
                exclude_3_atoms(words)
    for dihedral in args.dihedral:
        with open(dihedral) as f:
            f.readline()
            for line in f:
                words = line.split()
                exclude_4_atoms(words)

    for virtual in args.virtual:
        with open(virtual) as f:
            for line in f:
                words = line.split()
                t = int(words[0])
                if t == 0:
                    exclude_2_atoms(words[1:])
                elif t == 1:
                    exclude_3_atoms(words[1:])
                elif t in (2, 3):
                    exclude_4_atoms(words[1:])
                else:
                    raise Exception("virtual atom type wrong: are you sure this is a SPONGE virtual atom file?")

    for exclude in args.exclude:
        with open(exclude) as f:
            f.readline()
            count = 0
            for line in f:
                words = line.split()
                t = set(words[1:])
                partners[count] = partners[count].union(t)
                count += 1

    total = 0
    towrite = "{} {}\n"
    for i, p in enumerate(partners):
        newp = []
        for pi in p:
            if pi > i:
                newp.append(pi)
        towrite += "%d " % len(newp)
        towrite += ("{} " * len(newp)).format(*newp) + "\n"
        total += len(newp)
        towrite = towrite.format(args.n, total)

    f = Xopen(args.o, "w")
    f.write(towrite)
    f.close()


def name2name(args):
    """
    This **function** change the atom names from one file to another file

    :param args: arguments from argparse
    :return: None
    """
    from rdkit import Chem
    from rdkit.Chem import rdFMCS
    source("..")
    rdktool = source("..helper.rdkit")

    if args.to_format == "mol2":
        to_ = assign.Get_Assignment_From_Mol2(args.to_file)
    elif args.to_format == "gaff_mol2":
        source("..forcefield.amber.gaff")
        to_ = load_mol2(args.to_file).residues[0]
        to_ = assign.Get_Assignment_From_ResidueType(to_)
    elif args.to_format == "pdb":
        to_ = assign.Get_Assignment_From_PDB(args.to_file, determine_bond_order=False,
                                             only_residue=args.to_residue)

    if args.from_format == "mol2":
        from_ = assign.Get_Assignment_From_Mol2(args.from_file)
    elif args.from_format == "gaff_mol2":
        source("..forcefield.amber.gaff")
        from_ = load_mol2(args.from_file).residues[0]
        from_ = assign.Get_Assignment_From_ResidueType(from_)
    elif args.from_format == "pdb":
        from_ = assign.Get_Assignment_From_PDB(args.from_file, determine_bond_order=False,
                                               only_residue=args.from_residue)

    rdmol_a = rdktool.assign_to_rdmol(to_, True)
    rdmol_b = rdktool.assign_to_rdmol(from_, True)

    result = rdFMCS.FindMCS([rdmol_a, rdmol_b], completeRingsOnly=True, timeout=args.tmcs)
    rdmol_mcs = Chem.MolFromSmarts(result.smartsString)

    match_a = rdmol_a.GetSubstructMatch(rdmol_mcs)
    match_b = rdmol_b.GetSubstructMatch(rdmol_mcs)
    matchmap = {from_.names[match_b[j]]: to_.names[match_a[j]] for j in range(len(match_a))}
    from_.names = [matchmap.get(name, name) for name in from_.names]
    from_.name = args.out_residue

    if args.out_format == "mol2":
        from_.Save_As_Mol2(args.out_file)
    elif args.out_format == "pdb":
        from_.Save_As_PDB(args.out_file)
    elif args.out_format == "mcs_pdb":
        towrite = towrite = "REMARK   Generated By Xponge (Max Common Structure)\n"
        for i, atom in enumerate(from_.atoms):
            if i in match_b:
                towrite += "ATOM  %5d %4s %3s %1s%4d    %8.3f%8.3f%8.3f%17s%2s\n" % (i + 1, from_.names[i],
                                                                                     from_.name, " ", 1,
                                                                                     from_.coordinate[i][0],
                                                                                     from_.coordinate[i][1],
                                                                                     from_.coordinate[i][2], " ", atom)
        f = Xopen(args.out_file, "w")
        f.write(towrite)
        f.close()


def _mol2rfe_build(args, merged_from, merged_to):
    """

    :param args:
    :param merged_from:
    :param merged_to:
    :return:
    """
    source("..")
    fep = source("..forcefield.special.fep")
    min_ = source("..forcefield.special.min")

    if "build" in args.do:
        Xprint("\nBUILDING TOPOLOGY\n", verbose=-1)
        fep.Save_Soft_Core_LJ()

        for i in range(args.nl + 1):
            if os.path.exists("%d" % i):
                os.system("rm -rf %d" % i)
            os.mkdir("%d" % i)
            tt = fep.Merge_Force_Field(merged_from, merged_to, i / args.nl)
            if i == 0:
                min_.save_min_bonded_parameters()
            elif i == 1:
                min_.do_not_save_min_bonded_parameters()
            build.Save_SPONGE_Input(tt, "%d/%s" % (i, args.temp))


def _mol2rfe_output_path(subdir, workdir, tempname):
    """

    :param subdir:
    :param workdir:
    :param tempname:
    :return:
    """
    toadd = " -mdinfo {2}/{0}/{1}.mdinfo -mdout {2}/{0}/{1}.mdout".format(subdir, tempname, workdir)
    toadd += " -rst {2}/{0}/{1} -crd {2}/{0}/{1}.dat -box {2}/{0}/{1}.box".format(subdir, tempname, workdir)
    return toadd


def _mol2rfe_min(args):
    """

    :param args:
    :return:
    """
    source("..")

    if "min" in args.do:
        for i in range(args.nl + 1):
            if os.path.exists("%d/min" % i):
                os.system("rm -rf %d/min" % i)
            os.mkdir("%d/min" % i)
            basic = f"SPONGE -default_in_file_prefix {i}/{args.temp}"
            lambda_ = i / args.nl
            basic += f" -mode minimization -lambda_lj {lambda_}"
            basic += _mol2rfe_output_path("min", i, args.temp)
            if not args.mi:
                cif = " -constrain_mode SHAKE"
                run(f"{basic} {cif} -dt 1e-8 -step_limit {args.msteps[0]}")
                cif += " -coordinate_in_file {1}/min/{0}_coordinate.txt".format(args.temp, i)
                run(f"{basic} {cif} -dt 1e-7 -step_limit {args.msteps[1]}")
                run(f"{basic} {cif} -dt 1e-6 -step_limit {args.msteps[2]}")
                run(f"{basic} {cif} -dt 1e-5 -step_limit {args.msteps[3]}")
                run(f"{basic} {cif} -dt 1e-4 -step_limit {args.msteps[4]}")
                run(f"{basic} {cif} -dt 1e-3 -step_limit {args.msteps[5]}")
            else:
                mdin = args.mi.pop(0)
                cif = ""
                run(f"{basic} {cif} -mdin {mdin}")
                cif += " -coordinate_in_file {1}/min/{0}_coordinate.txt".format(args.temp, i)
                for mdin in args.mi:
                    run(f"{basic} {cif} -mdin {mdin}")


def _mol2rfe_pre_equilibrium(args):
    """

    :param args:
    :return:
    """
    source("..")

    if "pre_equilibrium" in args.do:
        for i in range(args.nl + 1):
            if os.path.exists("%d/pre_equilibrium" % i):
                os.system("rm -rf %d/pre_equilibrium" % i)
            os.mkdir("%d/pre_equilibrium" % i)
            command = f"SPONGE -default_in_file_prefix {i}/{args.temp}"
            lambda_ = i / args.nl
            command += f" -lambda_lj {lambda_}"
            command += _mol2rfe_output_path("pre_equilibrium", i, args.temp)
            command += f" -coordinate_in_file {i}/min/{args.temp}_coordinate.txt"
            if not args.pi:
                command += f" -mode NPT -step_limit {args.pre_equilibrium_step} -dt {args.dt} -constrain_mode SHAKE"
                command += f" -barostat {args.barostat} -thermostat {args.thermostat}"
                run(command)
            else:
                command += f" -mdin {args.pi}"
                run(command)


def _mol2rfe_equilibrium(args):
    """

    :param args:
    :return:
    """
    source("..")

    if "equilibrium" in args.do:
        for i in range(args.nl + 1):
            if os.path.exists("%d/equilibrium" % i):
                os.system("rm -rf %d/equilibrium" % i)
            os.mkdir("%d/equilibrium" % i)
            command = f"SPONGE -default_in_file_prefix {i}/{args.temp}"
            lambda_ = i / args.nl
            command += f" -lambda_lj {lambda_}"
            command += _mol2rfe_output_path("equilibrium", i, args.temp)
            command += f" -coordinate_in_file {i}/pre_equilibrium/{args.temp}_coordinate.txt"
            if not args.bi:
                command += f" -mode NPT -step_limit {args.equilibrium_step} -dt {args.dt} -constrain_mode SHAKE"
                command += f" -barostat {args.barostat} -thermostat {args.thermostat}"
                run(command)
            else:
                command += f" -mdin {args.pi}"
                run(command)


def _mol2rfe_analysis(args, merged_from):
    """

    :param args:
    :param merged_from:
    :return:
    """
    source("..")

    if "analysis" in args.do:
        f = Xopen("dh_dlambda.txt", "w")
        f.write("")
        f.close()
        if args.method == "TI":
            for i in range(args.nl + 1):
                if os.path.exists("%d/ti" % i):
                    os.system("rm -rf %d/ti" % i)
                os.mkdir("%d/ti" % i)
                inprefix = f"{i}/{args.temp}"
                command = f"SPONGE_TI -LJ_soft_core_in_file {inprefix}_LJ_soft_core.txt"
                command += " -exclude_in_file {0}_exclude.txt -charge_in_file {0}_charge.txt".format(inprefix)
                command += f" -chargeA_in_file 0/{args.temp}_charge.txt"
                command += f" -chargeB_in_file {args.nl}/{args.temp}_charge.txt"
                lambda_ = i / args.nl
                command += f" -lambda_lj {lambda_}"
                command += f" -subsys_division_in_file {inprefix}_subsys_division.txt  -charge_pertubated 1"
                inprefix = f"{i}/ti/{args.temp}"
                command += f" -mdinfo {inprefix}.mdinfo -mdout {inprefix}.mdout"
                inprefix = f"{i}/equilibrium/{args.temp}"
                command += f" -crd {inprefix}.dat -box {inprefix}.box -TI dh_dlambda.txt"
                command += f" -atom_numbers {len(merged_from.atoms)}"
                command += f" -frame_numbers {args.equilibrium // 100}"
                if not args.ai:
                    run(command)
                else:
                    command += f" -mdin {args.ai}"
                    run(command)
            dh_dlambda = np.loadtxt("dh_dlambda.txt")
            dh = []
            dh_int = []
            tempall = 0
            for i in range(args.nl):
                temp = dh_dlambda[i] * 0.5 / args.nl
                temp += dh_dlambda[i + 1] * 0.5 / args.nl
                dh.append(temp)
                tempall += temp
                dh_int.append(tempall)
            f = Xopen("free_energy.txt", "w")
            f.write("lambda_state\tFE(i+1)-FE(i)[kcal/mol]\tFE(i+1)-FE(0)[kcal/mol]\n")
            f.write("\n".join(["%d\t\t%.2f\t\t\t%.2f" % (i, dh[i], dh_int[i]) for i in range(args.nl)]))
            f.close()
        elif args.method == "FEP_BAR":
            raise NotImplementedError


def mol2rfe(args):
    """
    This **function** helps with the relative free energy calculation

    :param args: arguments from argparse
    :return: None
    """
    source("..")
    source("..forcefield.special.fep")
    source("..forcefield.special.min")

    if not args.ff:
        source("..forcefield.amber.gaff")
        source("..forcefield.amber.ff14sb")
        source("..forcefield.amber.tip3p")
    else:
        idic, ipy = os.path.split(args.ff)
        sys.path.append(idic)
        ipy, isuffix = os.path.splitext(ipy)
        assert isuffix == ".py", "the input force field file should be an Xponge file written by python"
        __import__(ipy)

    if not args.do:
        args.do = [["build", "min", "pre_equilibrium", "equilibrium", "analysis"]]
    args.do = args.do[0]

    from_res_type_ = load_mol2(args.r1).residues[0]
    from_ = assign.Get_Assignment_From_ResidueType(from_res_type_)
    if not args.ff:
        parmchk2_gaff(args.r1, args.temp + "_TMP1.frcmod")

    to_res_type_ = load_mol2(args.r2).residues[0]
    to_ = assign.Get_Assignment_From_ResidueType(to_res_type_)
    if not args.ff:
        parmchk2_gaff(args.r2, args.temp + "_TMP2.frcmod")

    for mol2file in args.r0:
        load_mol2(mol2file)

    rmol = load_pdb(args.pdb)

    merged_from, merged_to = Merge_Dual_Topology(rmol, rmol.residues[args.ri], to_res_type_, from_, to_, args.tmcs)

    if args.dohmr:
        H_Mass_Repartition(merged_from)
        H_Mass_Repartition(merged_to)

    _mol2rfe_build(args, merged_from, merged_to)

    _mol2rfe_min(args)

    _mol2rfe_pre_equilibrium(args)

    _mol2rfe_equilibrium(args)

    _mol2rfe_analysis(args, merged_from)
