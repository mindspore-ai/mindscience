"""
This **module** set the basic configuration for gaff
"""
from ...helper import source, Xprint, set_real_global_variable

source("....")
amber = source("...amber")

amber.load_parameters_from_parmdat("gaff.dat")

gaff = assign.AssignRule("gaff")


@gaff.Add_Rule("cx")
def _rule_cx(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    return assign.Atom_Judge(i, "C4") and "RG3" in assign.atom_marker[i].keys()


@gaff.Add_Rule("cy")
def _rule_cy(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    return assign.Atom_Judge(i, "C4") and "RG4" in assign.atom_marker[i].keys()


@gaff.Add_Rule("c3")
def _rule_c3(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    return assign.Atom_Judge(i, "C4")


@gaff.Add_Rule("c")
def _rule_c(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    tofind = assign.Atom_Judge(i, "C3")
    if tofind:
        tofind = False
        for bonded_atom in assign.bonds[i].keys():
            if assign.Atom_Judge(bonded_atom, ["O1", "S1"]):
                tofind = True
                break
    return tofind


@gaff.Add_Rule("cz")
def _rule_cz(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    tofind = assign.Atom_Judge(i, "C3")
    if tofind:
        for bonded_atom in assign.bonds[i].keys():
            if not assign.Atom_Judge(bonded_atom, "N3"):
                tofind = False
                break
    return tofind


@gaff.Add_Rule("cq")
def _rule_cq(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    tofind = assign.Atom_Judge(i, "C3") and "AR1" in assign.atom_marker[i] and assign.atom_marker[i].get("RG6") == 1
    for bonded_atom in assign.bonds[i].keys():
        if not tofind:
            break
        if assign.atoms[bonded_atom] not in assign.XX or "AR1" not in assign.atom_marker[bonded_atom].keys():
            tofind = False
    if tofind:
        tofind = False
        for bonded_atom in assign.bonds[i]:
            if tofind:
                break
            if (assign.atom_types[bonded_atom] == AtomType.get_type("cp") and \
                    "AB" in assign.bond_marker[bonded_atom][i]):
                tofind = True
                break
    return tofind


@gaff.Add_Rule("cp")
def _rule_cp(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    tofind = assign.Atom_Judge(i, "C3") and "AR1" in assign.atom_marker[i] and assign.atom_marker[i].get("RG6") == 1
    for bonded_atom in assign.bonds[i].keys():
        if not tofind:
            break
        if assign.atoms[bonded_atom] not in assign.XX or "AR1" not in assign.atom_marker[bonded_atom].keys():
            tofind = False

    return tofind


@gaff.Add_Rule("ca")
def _rule_ca(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    return assign.Atom_Judge(i, "C3") and "AR1" in assign.atom_marker[i]


def _single_double_name(i, assign, name1, name2):
    """

    :param i:
    :param assign:
    :param name1:
    :param name2:
    :return:
    """
    tofind = False
    for bonded_atom, bond_order in assign.bonds[i].items():
        if tofind:
            break
        if ((assign.atom_types[bonded_atom] == AtomType.get_type(name1) and bond_order == 2) or
                (assign.atom_types[bonded_atom] == AtomType.get_type(name2) and bond_order == 1)):
            tofind = True
            break
    return tofind


@gaff.Add_Rule("cd")
def _rule_cd(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    tofind = assign.Atom_Judge(i, "C3") and "sb" in assign.atom_marker[i] and "db" in assign.atom_marker[i] and \
             ("AR2" in assign.atom_marker[i] or "AR3" in assign.atom_marker[i])
    if tofind:
        tofind = _single_double_name(i, assign, "cc", "cd")
    return tofind


@gaff.Add_Rule("cc")
def _rule_cc(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    return assign.Atom_Judge(i, "C3") and "sb" in assign.atom_marker[i] and "db" in assign.atom_marker[i] and \
           ("AR2" in assign.atom_marker[i] or "AR3" in assign.atom_marker[i])


@gaff.Add_Rule("cf")
def _rule_cf(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    tofind = assign.Atom_Judge(i, "C3") and "sb" in assign.atom_marker[i] and "db" in assign.atom_marker[i]
    if tofind:
        tofind = False
        for bonded_atom, bond_order in assign.bonds[i].items():
            if tofind:
                break
            if bond_order == 1 and (assign.Atom_Judge(bonded_atom, ["C3", "C2", "N2", "P2"]) or
                                    (assign.Atom_Judge(bonded_atom, ["S3", "S4", "P3", "P4"]) and "db" in
                                     assign.atom_marker[bonded_atom])):
                tofind = True
                break
    if tofind:
        tofind = _single_double_name(i, assign, "ce", "cf")
    return tofind


@gaff.Add_Rule("ce")
def _rule_ce(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    tofind = assign.Atom_Judge(i, "C3") and "sb" in assign.atom_marker[i] and "db" in assign.atom_marker[i]
    if tofind:
        tofind = False
        for bonded_atom, bond_order in assign.bonds[i].items():
            if tofind:
                break
            if bond_order == 1 and (assign.Atom_Judge(bonded_atom, ["C3", "C2", "N2", "P2"]) or
                                    (assign.Atom_Judge(bonded_atom, ["S3", "S4", "P3", "P4"]) and "db" in
                                     assign.atom_marker[bonded_atom])):
                tofind = True
                break
    return tofind


@gaff.Add_Rule("cu")
def _rule_cu(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    return assign.Atom_Judge(i, "C3") and "RG3" in assign.atom_marker[i].keys()


@gaff.Add_Rule("cv")
def _rule_cv(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    return assign.Atom_Judge(i, "C3") and "RG4" in assign.atom_marker[i].keys()


@gaff.Add_Rule("c2")
def _rule_c2(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    return assign.Atom_Judge(i, "C3")


@gaff.Add_Rule("cg")
def _rule_cg(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    tofind = assign.Atom_Judge(i, "C2") and "sb" in assign.atom_marker[i] and "tb" in assign.atom_marker[i]
    if tofind:
        tofind = False
        for bonded_atom, bond_order in assign.bonds[i].items():
            if tofind:
                break
            if bond_order == 1 and assign.Atom_Judge(bonded_atom, ["C3", "C2", "N2", "P2", "N1"]):
                tofind = True
                break
    return tofind


@gaff.Add_Rule("c1")
def _rule_c1(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    return assign.Atom_Judge(i, "C2") or assign.Atom_Judge(i, "C1")


@gaff.Add_Rule("hn")
def _rule_hn(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    return assign.Atom_Judge(i, "H1") and assign.atoms[list(assign.bonds[i].keys())[0]] == "N"


@gaff.Add_Rule("ho")
def _rule_ho(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    return assign.Atom_Judge(i, "H1") and assign.atoms[list(assign.bonds[i].keys())[0]] == "O"


@gaff.Add_Rule("hs")
def _rule_hs(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    return assign.Atom_Judge(i, "H1") and assign.atoms[list(assign.bonds[i].keys())[0]] == "S"


@gaff.Add_Rule("hp")
def _rule_hp(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    return assign.Atom_Judge(i, "H1") and assign.atoms[list(assign.bonds[i].keys())[0]] == "P"


@gaff.Add_Rule("hx")
def _rule_hx(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    tofind = False
    for bonded_atom in assign.bonds[i].keys():
        if assign.atoms[bonded_atom] == "C":
            for bonded_atom_bonded in assign.bonds[bonded_atom].keys():
                if assign.Atom_Judge(bonded_atom_bonded, "N4"):
                    tofind = True
                    break
    return assign.Atom_Judge(i, "H1") and tofind


@gaff.Add_Rule("hw")
def _rule_hw(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    tofind = False
    for bonded_atom in assign.bonds[i].keys():
        if assign.atoms[bonded_atom] == "O":
            for bonded_atom_bonded in assign.bonds[bonded_atom].keys():
                if assign.Atom_Judge(bonded_atom_bonded, "H1"):
                    tofind = True
                    break
    return assign.Atom_Judge(i, "H1") and tofind


@gaff.Add_Rule("h3")
def _rule_h3(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    tofind = 0
    for bonded_atom in assign.bonds[i].keys():
        if assign.Atom_Judge(bonded_atom, "C4"):
            for bonded_atom_bonded in assign.bonds[bonded_atom].keys():
                if assign.atoms[bonded_atom_bonded] in assign.XE:
                    tofind += 1

    return assign.Atom_Judge(i, "H1") and tofind == 3


@gaff.Add_Rule("h2")
def _rule_h2(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    tofind = 0
    for bonded_atom in assign.bonds[i].keys():
        if assign.Atom_Judge(bonded_atom, "C4"):
            for bonded_atom_bonded in assign.bonds[bonded_atom].keys():
                if assign.atoms[bonded_atom_bonded] in assign.XE:
                    tofind += 1
    return assign.Atom_Judge(i, "H1") and tofind == 2


@gaff.Add_Rule("h1")
def _rule_h1(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    tofind = 0
    for bonded_atom in assign.bonds[i].keys():
        if assign.Atom_Judge(bonded_atom, "C4"):
            for bonded_atom_bonded in assign.bonds[bonded_atom].keys():
                if assign.atoms[bonded_atom_bonded] in assign.XE:
                    tofind += 1
    return assign.Atom_Judge(i, "H1") and tofind == 1


@gaff.Add_Rule("hc")
def _rule_hc(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    return assign.Atom_Judge(i, "H1") and assign.Atom_Judge(list(assign.bonds[i].keys())[0], "C4")


@gaff.Add_Rule("h5")
def _rule_h5(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    tofind = 0
    for bonded_atom in assign.bonds[i].keys():
        if assign.Atom_Judge(bonded_atom, "C3"):
            for bonded_atom_bonded in assign.bonds[bonded_atom].keys():
                if assign.atoms[bonded_atom_bonded] in assign.XE:
                    tofind += 1
    return assign.Atom_Judge(i, "H1") and tofind == 2


@gaff.Add_Rule("h4")
def _rule_h4(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    tofind = 0
    for bonded_atom in assign.bonds[i].keys():
        if assign.Atom_Judge(bonded_atom, "C3"):
            for bonded_atom_bonded in assign.bonds[bonded_atom].keys():
                if assign.atoms[bonded_atom_bonded] in assign.XE:
                    tofind += 1
    return assign.Atom_Judge(i, "H1") and tofind == 1


@gaff.Add_Rule("ha")
def _rule_ha(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    return assign.Atom_Judge(i, "H1")


@gaff.Add_Rule("f")
def _rule_f(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    return assign.atoms[i] == "F"


@gaff.Add_Rule("cl")
def _rule_cl(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    return assign.atoms[i] == "Cl"


@gaff.Add_Rule("br")
def _rule_br(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    return assign.atoms[i] == "Br"


@gaff.Add_Rule("i")
def _rule_i(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    return assign.atoms[i] == "I"


@gaff.Add_Rule("o")
def _rule_o(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    return assign.Atom_Judge(i, "O1")


@gaff.Add_Rule("oh")
def _rule_oh(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    tofind = False
    if assign.Atom_Judge(i, "O2") or assign.Atom_Judge(i, "O3"):
        for bonded_atom in assign.bonds[i].keys():
            if assign.Atom_Judge(bonded_atom, "H1"):
                tofind = True
                break
    return tofind


@gaff.Add_Rule("op")
def _rule_op(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    return assign.Atom_Judge(i, "O2") and "RG3" in assign.atom_marker[i].keys()


@gaff.Add_Rule("oq")
def _rule_oq(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    return assign.Atom_Judge(i, "O2") and "RG4" in assign.atom_marker[i].keys()


@gaff.Add_Rule("os")
def _rule_os(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    return assign.atoms[i] == "O"


@gaff.Add_Rule("ni")
def _rule_ni(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    tofind = False
    if assign.Atom_Judge(i, "N3") and "RG3" in assign.atom_marker[i].keys():
        for bonded_atom in assign.bonds[i].keys():
            if assign.Atom_Judge(bonded_atom, "C3"):
                for bonded_atom_bonded in assign.bonds[bonded_atom].keys():
                    if assign.Atom_Judge(bonded_atom_bonded, "O1") or assign.Atom_Judge(bonded_atom_bonded, "S1"):
                        tofind = True
                        break
            if tofind:
                break
    return tofind


@gaff.Add_Rule("nj")
def _rule_nj(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    tofind = False
    if assign.Atom_Judge(i, "N3") and "RG4" in assign.atom_marker[i].keys():
        for bonded_atom in assign.bonds[i].keys():
            if assign.Atom_Judge(bonded_atom, "C3"):
                for bonded_atom_bonded in assign.bonds[bonded_atom].keys():
                    if assign.Atom_Judge(bonded_atom_bonded, "O1") or assign.Atom_Judge(bonded_atom_bonded, "S1"):
                        tofind = True
                        break
            if tofind:
                break
    return tofind


@gaff.Add_Rule("n")
def _rule_n(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    tofind = False
    if assign.Atom_Judge(i, "N3"):
        for bonded_atom in assign.bonds[i].keys():
            if assign.Atom_Judge(bonded_atom, "C3"):
                for bonded_atom_bonded in assign.bonds[bonded_atom].keys():
                    if assign.Atom_Judge(bonded_atom_bonded, "O1") or assign.Atom_Judge(bonded_atom_bonded, "S1"):
                        tofind = True
                        break
            if tofind:
                break
    return tofind


@gaff.Add_Rule("nk")
def _rule_nk(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    return assign.Atom_Judge(i, "N4") and "RG3" in assign.atom_marker[i].keys()


@gaff.Add_Rule("nl")
def _rule_nl(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    return assign.Atom_Judge(i, "N4") and "RG4" in assign.atom_marker[i].keys()


@gaff.Add_Rule("n4")
def _rule_n4(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    return assign.Atom_Judge(i, "N4")


@gaff.Add_Rule("no")
def _rule_no(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    tofind = 0
    if assign.Atom_Judge(i, "N3"):
        for bonded_atom in assign.bonds[i].keys():
            if assign.Atom_Judge(bonded_atom, "O1"):
                tofind += 1
    return tofind == 2


@gaff.Add_Rule("na")
def _rule_na(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    return assign.Atom_Judge(i, "N3") and \
       ("AR1" in assign.atom_marker[i].keys() or "AR2" in assign.atom_marker[i].keys() or "AR3" in
        assign.atom_marker[i].keys())


@gaff.Add_Rule("nm")
def _rule_nm(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    tofind = False
    if assign.Atom_Judge(i, "N3") and "RG3" in assign.atom_marker[i].keys():
        for bonded_atom in assign.bonds[i].keys():
            if ("DB" in assign.atom_marker[bonded_atom].keys() and
                    (assign.Atom_Judge(bonded_atom, "C3") or
                     assign.Atom_Judge(bonded_atom, "N2") or
                     assign.Atom_Judge(bonded_atom, "P2"))):
                tofind = True
                break
            if (("AR1" in assign.atom_marker[bonded_atom].keys() or
                 "AR2" in assign.atom_marker[bonded_atom].keys() or
                 "AR3" in assign.atom_marker[bonded_atom].keys()) and
                    assign.atoms[bonded_atom] in assign.XX):
                tofind = True
                break
    return tofind


@gaff.Add_Rule("nn")
def _rule_nn(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    tofind = False
    if assign.Atom_Judge(i, "N3") and "RG4" in assign.atom_marker[i].keys():
        for bonded_atom in assign.bonds[i].keys():
            if "DB" in assign.atom_marker[bonded_atom].keys() and assign.Atom_Judge(bonded_atom,
                                                                                    ["C3", "N2", "P2"]):
                tofind = True
                break
            if (("AR1" in assign.atom_marker[bonded_atom].keys() or
                 "AR2" in assign.atom_marker[bonded_atom].keys() or
                 "AR3" in assign.atom_marker[bonded_atom].keys()) and
                    assign.atoms[bonded_atom] in assign.XX):
                tofind = True
                break
    return tofind


@gaff.Add_Rule("nh")
def _rule_nh(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    tofind = False
    if assign.Atom_Judge(i, "N3"):
        for bonded_atom in assign.bonds[i].keys():
            if "DB" in assign.atom_marker[bonded_atom].keys() and assign.Atom_Judge(bonded_atom,
                                                                                    ["C3", "N2", "P2"]):
                tofind = True
                break
            if (("AR1" in assign.atom_marker[bonded_atom].keys() or
                 "AR2" in assign.atom_marker[bonded_atom].keys() or
                 "AR3" in assign.atom_marker[bonded_atom].keys()) and
                    assign.atoms[bonded_atom] in assign.XX):
                tofind = True
                break
    return tofind


@gaff.Add_Rule("np")
def _rule_np(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    return assign.Atom_Judge(i, "N3") and "RG3" in assign.atom_marker[i].keys()


@gaff.Add_Rule("nq")
def _rule_nq(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    return assign.Atom_Judge(i, "N3") and "RG4" in assign.atom_marker[i].keys()


@gaff.Add_Rule("n3")
def _rule_n3(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    return assign.Atom_Judge(i, "N3")


@gaff.Add_Rule("nb")
def _rule_nb(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    return assign.Atom_Judge(i, "N2") and "AR1" in assign.atom_marker[i].keys()


@gaff.Add_Rule("nd")
def _rule_nd(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    tofind = assign.Atom_Judge(i, "N2") and "sb" in assign.atom_marker[i] and "db" in assign.atom_marker[i] and \
         ("AR2" in assign.atom_marker[i] or "AR3" in assign.atom_marker[i])
    if tofind:
        tofind = False
        for bonded_atom, bond_order in assign.bonds[i].items():
            if bond_order == 1 and (assign.Atom_Judge(bonded_atom, ["C3", "C2", "N2", "P2"]) or
                                    (assign.Atom_Judge(bonded_atom, ["S3", "S4", "P3", "P4"]) and "db" in
                                     assign.atom_marker[bonded_atom])):
                tofind = True
                break
            if assign.Atom_Judge(bonded_atom, ["C3", "C2", "N2", "P2"]):
                for bonded_atom_bonded in assign.bonds[bonded_atom].keys():
                    if assign.Atom_Judge(bonded_atom_bonded, ["C3", "C2", "N2", "P2"]):
                        tofind = True
                        break
                if tofind:
                    break
    if tofind:
        tofind = _single_double_name(i, assign, "nc", "nd")
    return tofind


@gaff.Add_Rule("nc")
def _rule_nc(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    tofind = assign.Atom_Judge(i, "N2") and "sb" in assign.atom_marker[i] and "db" in assign.atom_marker[i] and \
         ("AR2" in assign.atom_marker[i] or "AR3" in assign.atom_marker[i])
    if tofind:
        tofind = False
        for bonded_atom, bond_order in assign.bonds[i].items():
            if bond_order == 1 and (assign.Atom_Judge(bonded_atom, ["C3", "C2", "N2", "P2"]) or
                                    (assign.Atom_Judge(bonded_atom, ["S3", "S4", "P3", "P4"]) and "db" in
                                     assign.atom_marker[bonded_atom])):
                tofind = True
                break
            if assign.Atom_Judge(bonded_atom, ["C3", "C2", "N2", "P2"]):
                for bonded_atom_bonded in assign.bonds[bonded_atom].keys():
                    if assign.Atom_Judge(bonded_atom_bonded, ["C3", "C2", "N2", "P2"]):
                        tofind = True
                        break
                if tofind:
                    break
    return tofind


@gaff.Add_Rule("nf")
def _rule_nf(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    tofind = assign.Atom_Judge(i, "N2") and "sb" in assign.atom_marker[i] and "db" in assign.atom_marker[i]
    if tofind:
        tofind = False
        for bonded_atom, bond_order in assign.bonds[i].items():
            if bond_order == 1 and (assign.Atom_Judge(bonded_atom, ["C3", "C2", "N2", "P2"]) or
                                    (assign.Atom_Judge(bonded_atom, ["S3", "S4", "P3", "P4"]) and "db" in
                                     assign.atom_marker[bonded_atom])):
                tofind = True
                break
    if tofind:
        tofind = _single_double_name(i, assign, "ne", "nf")
    return tofind


@gaff.Add_Rule("ne")
def _rule_ne(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    tofind = assign.Atom_Judge(i, "N2") and "sb" in assign.atom_marker[i] and "db" in assign.atom_marker[i]
    if tofind:
        tofind = False
        for bonded_atom, bond_order in assign.bonds[i].items():
            if bond_order == 1 and (assign.Atom_Judge(bonded_atom, ["C3", "C2", "N2", "P2"]) or
                                    (assign.Atom_Judge(bonded_atom, ["S3", "S4", "P3", "P4"]) and "db" in
                                     assign.atom_marker[bonded_atom])):
                tofind = True
                break
    return tofind


@gaff.Add_Rule("n1")
def _rule_n1(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    return assign.Atom_Judge(i, "N1") or (assign.Atom_Judge(i, "N2") and
                                          (("sb" in assign.atom_marker[i] and "tb" in assign.atom_marker[i]) or
                                           (assign.atom_marker[i].get("db") == 2)))


@gaff.Add_Rule("n2")
def _rule_n2(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    return assign.Atom_Judge(i, "N2")


@gaff.Add_Rule("s")
def _rule_s(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    return assign.Atom_Judge(i, "S1")


@gaff.Add_Rule("s2")
def _rule_s2(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    return assign.Atom_Judge(i, "S2") and ("DB" in assign.atom_marker[i] or "TB" in assign.atom_marker[i])


@gaff.Add_Rule("sh")
def _rule_sh(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    tofind = False
    if assign.Atom_Judge(i, "S2"):
        for bonded_atom in assign.bonds[i].keys():
            if assign.Atom_Judge(bonded_atom, "H1"):
                tofind = True
                break
    return tofind


@gaff.Add_Rule("sp")
def _rule_sp(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    return assign.Atom_Judge(i, "S2") and "RG3" in assign.atom_marker[i].keys()


@gaff.Add_Rule("sq")
def _rule_sq(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    return assign.Atom_Judge(i, "S2") and "RG4" in assign.atom_marker[i].keys()


@gaff.Add_Rule("ss")
def _rule_ss(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    return assign.Atom_Judge(i, "S2")


@gaff.Add_Rule("sx")
def _rule_sx(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    tofind = assign.Atom_Judge(i, "S3") and "db" in assign.atom_marker[i].keys()
    if tofind:
        tofind = False
        for bonded_atom, bond_order in assign.bonds[i].items():
            if bond_order == 1 and (assign.Atom_Judge(bonded_atom, ["C3", "N2", "P2"]) or
                                    (assign.Atom_Judge(bonded_atom, ["S3", "S4", "P3", "P4"]) and "db" in
                                     assign.atom_marker[bonded_atom])):
                tofind = True
                break
    return tofind


@gaff.Add_Rule("s4")
def _rule_s4(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    return assign.Atom_Judge(i, "S3")


@gaff.Add_Rule("sy")
def _rule_sy(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    tofind = assign.Atom_Judge(i, "S4") and "db" in assign.atom_marker[i].keys()
    if tofind:
        tofind = False
        for bonded_atom, bond_order in assign.bonds[i].items():
            if bond_order == 1 and (assign.Atom_Judge(bonded_atom, ["C3", "N2", "P2"]) or
                                    (assign.Atom_Judge(bonded_atom, ["S3", "S4", "P3", "P4"]) and "db" in
                                     assign.atom_marker[bonded_atom])):
                tofind = True
                break
    return tofind


@gaff.Add_Rule("s6")
def _rule_s6(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    return assign.Atom_Judge(i, "S4") or assign.Atom_Judge(i, "S5") or assign.Atom_Judge(i, "S6")


@gaff.Add_Rule("pd")
def _rule_pd(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    tofind = assign.Atom_Judge(i, "P2") and "sb" in assign.atom_marker[i] and "db" in assign.atom_marker[i] and \
             ("AR2" in assign.atom_marker[i] or "AR3" in assign.atom_marker[i])
    if tofind:
        tofind = False
        for bonded_atom, bond_order in assign.bonds[i].items():
            if bond_order == 1 and (assign.Atom_Judge(bonded_atom, ["C3", "N2", "P2"]) or
                                    (assign.Atom_Judge(bonded_atom, ["S3", "S4", "P3", "P4"]) and "db" in
                                     assign.atom_marker[bonded_atom])):
                tofind = True
                break
    if tofind:
        tofind = _single_double_name(i, assign, "pc", "pd")
    return tofind


@gaff.Add_Rule("pb")
def _rule_pb(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    return assign.Atom_Judge(i, "P2") and "AR1" in assign.atom_marker[i].keys()


@gaff.Add_Rule("pc")
def _rule_pc(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    tofind = assign.Atom_Judge(i, "P2") and "sb" in assign.atom_marker[i] and "db" in assign.atom_marker[i] and \
         ("AR2" in assign.atom_marker[i] or "AR3" in assign.atom_marker[i])
    if tofind:
        tofind = False
        for bonded_atom, bond_order in assign.bonds[i].items():
            if bond_order == 1 and (assign.Atom_Judge(bonded_atom, ["C3", "N2", "P2"]) or
                                    (assign.Atom_Judge(bonded_atom, ["S3", "S4", "P3", "P4"]) and "db" in
                                     assign.atom_marker[bonded_atom])):
                tofind = True
                break
    return tofind


@gaff.Add_Rule("pf")
def _rule_pf(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    tofind = assign.Atom_Judge(i, "P2") and "sb" in assign.atom_marker[i] and "db" in assign.atom_marker[i]
    if tofind:
        tofind = False
        for bonded_atom, bond_order in assign.bonds[i].items():
            if bond_order == 1 and (assign.Atom_Judge(bonded_atom, ["C3", "N2", "P2"]) or
                                    (assign.Atom_Judge(bonded_atom, ["S3", "S4", "P3", "P4"]) and "db" in
                                     assign.atom_marker[bonded_atom])):
                tofind = True
                break
    if tofind:
        tofind = _single_double_name(i, assign, "pe", "pf")
    return tofind


@gaff.Add_Rule("pe")
def _rule_pe(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    tofind = assign.Atom_Judge(i, "P2") and "sb" in assign.atom_marker[i] and "db" in assign.atom_marker[i]
    if tofind:
        tofind = False
        for bonded_atom, bond_order in assign.bonds[i].items():
            if bond_order == 1 and (assign.Atom_Judge(bonded_atom, ["C3", "N2", "P2"]) or
                                    (assign.Atom_Judge(bonded_atom, ["S3", "S4", "P3", "P4"]) and "db" in
                                     assign.atom_marker[bonded_atom])):
                tofind = True
                break
    return tofind


@gaff.Add_Rule("p2")
def _rule_p2(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    return assign.Atom_Judge(i, "P1") or assign.Atom_Judge(i, "P2")


@gaff.Add_Rule("px")
def _rule_px(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    tofind = False
    if assign.Atom_Judge(i, "P3") and "db" in assign.atom_marker[i].keys():
        for bonded_atom, bond_order in assign.bonds[i].items():
            if bond_order == 1 and (assign.Atom_Judge(bonded_atom, ["C3", "N2", "P2"]) or
                                    (assign.Atom_Judge(bonded_atom, ["S3", "S4", "P3", "P4"]) and "db" in
                                     assign.atom_marker[bonded_atom])):
                tofind = True
                break
    return tofind


@gaff.Add_Rule("p4")
def _rule_p4(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    tofind = False
    if assign.Atom_Judge(i, "P3") and "db" in assign.atom_marker[i].keys():
        for bonded_atom in assign.bonds[i].keys():
            if assign.Atom_Judge(bonded_atom, "O1") or assign.Atom_Judge(bonded_atom, "S1"):
                tofind = True
                break
    return tofind


@gaff.Add_Rule("p3")
def _rule_p3(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    return assign.Atom_Judge(i, "P3")


@gaff.Add_Rule("py")
def _rule_py(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    tofind = False
    if assign.Atom_Judge(i, "P4") and "db" in assign.atom_marker[i].keys():
        for bonded_atom, bond_order in assign.bonds[i].items():
            if bond_order == 1 and (assign.Atom_Judge(bonded_atom, ["C3", "N2", "P2"]) or
                                    (assign.Atom_Judge(bonded_atom, ["S3", "S4", "P3", "P4"]) and "db" in
                                     assign.atom_marker[bonded_atom])):
                tofind = True
                break
    return tofind


@gaff.Add_Rule("p5")
def _rule_p5(i, assign):
    """

    :param i:
    :param assign:
    :return:
    """
    return assign.Atom_Judge(i, "P4") or assign.Atom_Judge(i, "P5") or assign.Atom_Judge(i, "P6")


def parmchk2_gaff(ifname, ofname, direct_load=True, keep=True):
    """
    This **function** is to do parmchk2 for gaff

    :param ifname: a string of mol2 file name, a ResidueType, Residue or Molecule instance
    :param ofname: the output file name
    :param direct_load: directly load the output file after writing it
    :param keep: do not delete the output file after loading it
    :return: None
    """
    import XpongeLib as xlib
    datapath = os.path.split(xlib.__file__)[0]
    if isinstance(ifname, AbstractMolecule):
        Save_Mol2(ifname, "temp.mol2")
        ifname = "temp.mol2"
    parmchk2_func = getattr(xlib, "_parmchk2")
    parmchk2_func(ifname, "mol2", ofname, datapath, 0, 1, 1)
    if direct_load:
        amber.load_parameters_from_frcmod(ofname, prefix=False)
    if not keep:
        os.remove(ofname)
    if os.path.exists("temp.mol2"):
        os.remove("temp.mol2")


set_real_global_variable("parmchk2_gaff", parmchk2_gaff)

Xprint("""Reference for gaff:
  Wang, J., Wolf, R.M., Caldwell, J.W., Kollman, P.A. and Case, D.A.
    Development and testing of a general amber force field.
    Journal of Computational Chemistry 2004 25, 1157-1174
    DOI: 10.1002/jcc.20035
""")
# pylint:disable=undefined-variable
