# Modified from RFdiffusion (https://github.com/RosettaCommons/RFdiffusion)
# Original license: BSD License
#
# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


import mindspore as ms
from mindspore import ops


class Potential:
    """
    Interface class that defines the functions a potential must implement
    """

    def compute(self, xyz):
        """
        Given the current structure of the model prediction, return the current
        potential as a MindSpore tensor with a single entry

        Args:
            xyz (mindspore.tensor, size: [L,27,3]: The current coordinates of the sample

        Returns:
            potential (mindspore.tensor, size: [1]): A potential whose value will be MAXIMIZED
                                                 by taking a step along it's gradient
        """
        raise NotImplementedError("Potential compute function was not overwritten")


class monomer_ROG(Potential):
    """
    Radius of Gyration potential for encouraging monomer compactness

    Written by DJ and refactored into a class by NRB
    """

    def __init__(self, weight=1, min_dist=15):

        self.weight = weight
        self.min_dist = min_dist

    def compute(self, xyz):
        Ca = xyz[:, 1]  # [L,3]

        centroid = ms.mint.mean(Ca, dim=0, keepdim=True)  # [1,3]

        diff = Ca - centroid  # [1, L, 1, 3]
        d2 = ms.mint.sum(diff * diff, dim=-1)  # [1, L, 1]
        dgram = ops.sqrt(d2)  # [1, L, 1]
        dgram = ms.mint.maximum(
            self.min_dist * ms.mint.ones_like(dgram), dgram
        )  # [L,1,3]

        rad_of_gyration = ms.mint.sqrt(
            ms.mint.sum(ms.mint.square(dgram)) / Ca.shape[0]
        )  # [1]

        return -1 * self.weight * rad_of_gyration


class binder_ROG(Potential):
    """
    Radius of Gyration potential for encouraging binder compactness

    Author: NRB
    """

    def __init__(self, binderlen, weight=1, min_dist=15):

        self.binderlen = binderlen
        self.min_dist = min_dist
        self.weight = weight

    def compute(self, xyz):

        # Only look at binder residues
        Ca = xyz[: self.binderlen, 1]  # [Lb,3]

        centroid = ms.mint.mean(Ca, dim=0, keepdim=True)  # [1,3]

        diff = Ca - centroid  # [1, L, 1, 3]
        d2 = ms.mint.sum(diff * diff, dim=-1)  # [1, L, 1]
        dgram = ops.sqrt(d2)  # [1, L, 1]

        dgram = ms.mint.maximum(
            self.min_dist * ms.mint.ones_like(dgram.squeeze(0)), dgram.squeeze(0)
        )  # [Lb,1,3]

        rad_of_gyration = ms.mint.sqrt(
            ms.mint.sum(ms.mint.square(dgram)) / Ca.shape[0]
        )  # [1]

        return -1 * self.weight * rad_of_gyration


class dimer_ROG(Potential):
    """
    Radius of Gyration potential for encouraging compactness of both monomers when designing dimers

    Author: PV
    """

    def __init__(self, binderlen, weight=1, min_dist=15):

        self.binderlen = binderlen
        self.min_dist = min_dist
        self.weight = weight

    def compute(self, xyz):

        # Only look at monomer 1 residues
        Ca_m1 = xyz[: self.binderlen, 1]  # [Lb,3]

        # Only look at monomer 2 residues
        Ca_m2 = xyz[self.binderlen :, 1]  # [Lb,3]

        centroid_m1 = ms.mint.mean(Ca_m1, dim=0, keepdim=True)  # [1,3]
        centroid_m2 = ms.mint.mean(Ca_m1, dim=0, keepdim=True)  # [1,3]

        # This calculates RoG for Monomer 1
        diff = Ca_m1 - centroid_m1  # [1, L, 1, 3]
        d2 = ms.mint.sum(diff * diff, dim=-1)  # [1, L, 1]
        dgram_m1 = ops.sqrt(d2)  # [1, L, 1]
        dgram_m1 = ms.mint.maximum(
            self.min_dist * ms.mint.ones_like(dgram_m1.squeeze(0)), dgram_m1.squeeze(0)
        )  # [Lb,1,3]
        rad_of_gyration_m1 = ms.mint.sqrt(
            ms.mint.sum(ms.mint.square(dgram_m1)) / Ca_m1.shape[0]
        )  # [1]

        # This calculates RoG for Monomer 2
        diff = Ca_m2 - centroid_m2  # [1, L, 1, 3]
        d2 = ms.mint.sum(diff * diff, dim=-1)  # [1, L, 1]
        dgram_m2 = ops.sqrt(d2)  # [1, L, 1]
        dgram_m2 = ms.mint.maximum(
            self.min_dist * ms.mint.ones_like(dgram_m2.squeeze(0)), dgram_m2.squeeze(0)
        )  # [Lb,1,3]
        rad_of_gyration_m2 = ms.mint.sqrt(
            ms.mint.sum(ms.mint.square(dgram_m2)) / Ca_m2.shape[0]
        )  # [1]

        # Potential value is the average of both radii of gyration (is avg. the best way to do this?)
        return -1 * self.weight * (rad_of_gyration_m1 + rad_of_gyration_m2) / 2


class binder_ncontacts(Potential):
    """
    Differentiable way to maximise number of contacts within a protein

    """

    def __init__(self, binderlen, weight=1, r_0=8, d_0=4):

        self.binderlen = binderlen
        self.r_0 = r_0
        self.weight = weight
        self.d_0 = d_0

    def compute(self, xyz):

        # Only look at binder Ca residues
        Ca = xyz[: self.binderlen, 1]  # [Lb,3]

        # Ca: [Lb, 3]
        Ca_exp = Ca[None, :, :]  # [1, Lb, 3]

        x2 = ms.mint.sum(Ca_exp * Ca_exp, dim=-1, keepdim=True)  # [1, Lb, 1]
        y2 = ms.mint.sum(Ca_exp * Ca_exp, dim=-1, keepdim=True).swapaxes(
            -2, -1
        )  # [1, 1, Lb]
        cross = ms.mint.matmul(Ca_exp, Ca_exp.swapaxes(-2, -1))  # [1, Lb, Lb]

        d2 = ops.relu(x2 + y2 - 2.0 * cross)  # [1, Lb, Lb]
        dgram = ms.mint.sqrt(d2)  # [1, Lb, Lb]

        divide_by_r_0 = (dgram - self.d_0) / self.r_0
        numerator = ms.mint.pow(divide_by_r_0, 6)
        denominator = ms.mint.pow(divide_by_r_0, 12)
        binder_ncontacts = (1 - numerator) / (1 - denominator)

        print("BINDER CONTACTS:", binder_ncontacts.sum())
        # Potential value is the average of both radii of gyration (is avg. the best way to do this?)
        return self.weight * binder_ncontacts.sum()


class interface_ncontacts(Potential):
    """
    Differentiable way to maximise number of contacts between binder and target

    Author: PV
    """

    def __init__(self, binderlen, weight=1, r_0=8, d_0=6):

        self.binderlen = binderlen
        self.r_0 = r_0
        self.weight = weight
        self.d_0 = d_0

    def compute(self, xyz):

        # Extract binder Ca residues
        Ca_b = xyz[: self.binderlen, 1]  # [Lb,3]

        # Extract target Ca residues
        Ca_t = xyz[self.binderlen :, 1]  # [Lt,3]

        Ca_b_exp = Ca_b[None, :, :]  # [1, Lb, 3]
        Ca_t_exp = Ca_t[None, :, :]  # [1, Lt, 3]

        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
        x2 = ms.mint.sum(Ca_b_exp * Ca_b_exp, dim=-1, keepdim=True)  # [1, Lb, 1]
        y2 = ms.mint.sum(Ca_t_exp * Ca_t_exp, dim=-1, keepdim=True).swapaxes(
            -2, -1
        )  # [1, 1, Lt]
        cross = ms.mint.matmul(Ca_b_exp, Ca_t_exp.swapaxes(-2, -1))  # [1, Lb, Lt]

        d2 = ops.relu(x2 + y2 - 2.0 * cross)
        dgram = ms.mint.sqrt(d2)  # [1, Lb, Lt]

        divide_by_r_0 = (dgram - self.d_0) / self.r_0
        numerator = ms.mint.pow(divide_by_r_0, 6)
        denominator = ms.mint.pow(divide_by_r_0, 12)
        interface_ncontacts = (1 - numerator) / (1 - denominator)
        # Potential is the sum of values in the tensor
        interface_ncontacts = interface_ncontacts.sum()

        print("INTERFACE CONTACTS:", interface_ncontacts.sum())

        return self.weight * interface_ncontacts


class monomer_contacts(Potential):
    """
    Differentiable way to maximise number of contacts within a protein

    Author: PV

    NOTE: This function sometimes produces NaN's -- added check in reverse diffusion for nan grads
    """

    def __init__(self, weight=1, r_0=8, d_0=2, eps=1e-6):

        self.r_0 = r_0
        self.weight = weight
        self.d_0 = d_0
        self.eps = eps

    def compute(self, xyz):

        Ca = xyz[:, 1]  # [L,3]

        # Ca: [Lb, 3]
        Ca_exp = Ca[None, :, :]  # [1, Lb, 3]

        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2x·y
        x2 = ms.mint.sum(Ca_exp * Ca_exp, dim=-1, keepdim=True)  # [1, Lb, 1]
        y2 = ms.mint.sum(Ca_exp * Ca_exp, dim=-1, keepdim=True).swapaxes(
            -2, -1
        )  # [1, 1, Lb]
        cross = ms.mint.matmul(Ca_exp, Ca_exp.swapaxes(-2, -1))  # [1, Lb, Lb]

        d2 = ops.relu(x2 + y2 - 2.0 * cross)  # [1, Lb, Lb]
        dgram = ms.mint.sqrt(d2)  # [1, Lb, Lb]

        divide_by_r_0 = (dgram - self.d_0) / self.r_0
        numerator = ms.mint.pow(divide_by_r_0, 6)
        denominator = ms.mint.pow(divide_by_r_0, 12)

        ncontacts = (1 - numerator) / ((1 - denominator))

        # Potential value is the average of both radii of gyration (is avg. the best way to do this?)
        return self.weight * ncontacts.sum()


class olig_contacts(Potential):
    """
    Applies PV's num contacts potential within/between chains in symmetric oligomers

    Author: DJ
    """

    def __init__(self, contact_matrix, weight_intra=1, weight_inter=1, r_0=8, d_0=2):
        """
        Parameters:
            chain_lengths (list, required): List of chain lengths, length is (Nchains)

            contact_matrix (mindspore.tensor/np.array, required):
                square matrix of shape (Nchains,Nchains) whose (i,j) enry represents
                attractive (1), repulsive (-1), or non-existent (0) contact potentials
                between chains in the complex

            weight (int/float, optional): Scaling/weighting factor
        """
        self.contact_matrix = contact_matrix
        self.weight_intra = weight_intra
        self.weight_inter = weight_inter
        self.r_0 = r_0
        self.d_0 = d_0

        # check contact matrix only contains valid entries
        assert all(
            [i in [-1, 0, 1] for i in contact_matrix.flatten()]
        ), "Contact matrix must contain only 0, 1, or -1 in entries"
        # assert the matrix is square and symmetric
        shape = contact_matrix.shape
        assert len(shape) == 2
        assert shape[0] == shape[1]
        for i in range(shape[0]):
            for j in range(shape[1]):
                assert contact_matrix[i, j] == contact_matrix[j, i]
        self.nchain = shape[0]

    def _get_idx(self, i, L):
        """
        Returns the zero-indexed indices of the residues in chain i
        """
        assert L % self.nchain == 0
        Lchain = L // self.nchain
        return i * Lchain + ms.mint.arange(Lchain)

    def compute(self, xyz):
        """
        Iterate through the contact matrix, compute contact potentials between chains that need it,
        and negate contacts for any
        """
        L = xyz.shape[0]

        all_contacts = 0
        for i in range(self.nchain):
            for j in range(self.nchain):
                # only compute for upper triangle, disregard zeros in contact matrix
                if (i <= j) and (self.contact_matrix[i, j] != 0):

                    # get the indices for these two chains
                    idx_i = self._get_idx(i, L)
                    idx_j = self._get_idx(j, L)

                    Ca_i = xyz[idx_i, 1]  # slice out crds for this chain
                    Ca_j = xyz[idx_j, 1]  # slice out crds for that chain

                    Ca_i_exp = Ca_i[None, :, :]  # [1, Lb, 3]
                    Ca_j_exp = Ca_j[None, :, :]  # [1, Lt, 3]

                    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
                    x2 = ms.mint.sum(
                        Ca_i_exp * Ca_i_exp, dim=-1, keepdim=True
                    )  # [1, Lb, 1]
                    y2 = ms.mint.sum(
                        Ca_j_exp * Ca_j_exp, dim=-1, keepdim=True
                    ).swapaxes(
                        -2, -1
                    )  # [1, 1, Lt]
                    cross = ms.mint.matmul(
                        Ca_i_exp, Ca_j_exp.swapaxes(-2, -1)
                    )  # [1, Lb, Lt]

                    d2 = ops.relu(x2 + y2 - 2.0 * cross)
                    dgram = ms.mint.sqrt(d2)  # [1, Lb, Lt]

                    divide_by_r_0 = (dgram - self.d_0) / self.r_0
                    numerator = ms.mint.pow(divide_by_r_0, 6)
                    denominator = ms.mint.pow(divide_by_r_0, 12)
                    ncontacts = (1 - numerator) / (1 - denominator)

                    # weight, don't double count intra
                    scalar = (i == j) * self.weight_intra / 2 + (
                        i != j
                    ) * self.weight_inter

                    #                 contacts              attr/repuls          relative weights
                    all_contacts += ncontacts.sum() * self.contact_matrix[i, j] * scalar

        return all_contacts


def get_damped_lj(r_min, r_lin, p1=6, p2=12):

    y_at_r_lin = lj(r_lin, r_min, p1, p2)
    ydot_at_r_lin = lj_grad(r_lin, r_min, p1, p2)

    def inner(dgram):
        return (dgram < r_lin) * (ydot_at_r_lin * (dgram - r_lin) + y_at_r_lin) + (
            dgram >= r_lin
        ) * lj(dgram, r_min, p1, p2)

    return inner


def lj(dgram, r_min, p1=6, p2=12):
    return 4 * (
        (r_min / (2 ** (1 / p1) * dgram)) ** p2
        - (r_min / (2 ** (1 / p1) * dgram)) ** p1
    )


def lj_grad(dgram, r_min, p1=6, p2=12):
    return -p2 * r_min**p1 * (r_min**p1 - dgram**p1) / (dgram ** (p2 + 1))


def mask_expand(mask, n=1):
    mask_out = mask.clone()
    assert mask.ndim == 1
    for i in ms.mint.where(mask)[0]:
        for j in range(i - n, i + n + 1):
            if j >= 0 and j < len(mask):
                mask_out[j] = True
    return mask_out


def contact_energy(dgram, d_0, r_0):
    divide_by_r_0 = (dgram - d_0) / r_0
    numerator = ms.mint.pow(divide_by_r_0, 6)
    denominator = ms.mint.pow(divide_by_r_0, 12)

    ncontacts = (1 - numerator) / ((1 - denominator)).float()
    return -ncontacts


def poly_repulse(dgram, r, slope, p=1):
    a = slope / (p * r ** (p - 1))

    return (dgram < r) * a * ms.mint.abs(r - dgram) ** p * slope


class substrate_contacts(Potential):
    """
    Implicitly models a ligand with an attractive-repulsive potential.
    """

    def __init__(
        self, weight=1, r_0=8, d_0=2, s=1, eps=1e-6, rep_r_0=5, rep_s=2, rep_r_min=1
    ):

        self.r_0 = r_0
        self.weight = weight
        self.d_0 = d_0
        self.eps = eps

        # motif frame coordinates
        # NOTE: these probably need to be set after sample_init() call, because the motif sequence position in design must be known
        # [4,3] xyz coordinates from 4 atoms of input motif
        self.motif_frame = None
        # list of tuples giving positions of above atoms in design [(resi, atom_idx)]
        self.motif_mapping = None
        self.motif_substrate_atoms = (
            None  # xyz coordinates of substrate from input motif
        )
        r_min = 2
        self.energies = []
        self.energies.append(
            lambda dgram: s * contact_energy(ms.mint.min(dgram, dim=-1)[0], d_0, r_0)
        )
        if rep_r_min:
            self.energies.append(
                lambda dgram: poly_repulse(
                    ms.mint.min(dgram, dim=-1)[0], rep_r_0, rep_s, p=1.5
                )
            )
        else:
            self.energies.append(
                lambda dgram: poly_repulse(dgram, rep_r_0, rep_s, p=1.5)
            )

    def compute(self, xyz):

        # First, get random set of atoms
        # This operates on self.xyz_motif, which is assigned to this class in the model runner (for horrible plumbing reasons)
        self._grab_motif_residues(self.xyz_motif)

        # for checking affine transformation is correct
        first_distance = ms.mint.sqrt(
            ms.mint.sqrt(
                ms.mint.sum(
                    ms.mint.square(self.motif_substrate_atoms[0] - self.motif_frame[0]),
                    dim=-1,
                )
            )
        )

        # grab the coordinates of the corresponding atoms in the new frame using mapping
        res = ms.tensor([k[0] for k in self.motif_mapping])
        atoms = ms.tensor([k[1] for k in self.motif_mapping])
        new_frame = xyz[self.diffusion_mask][res.T, atoms, :].squeeze(0)
        # calculate affine transformation matrix and translation vector b/w new frame and motif frame
        A, t = self._recover_affine(self.motif_frame, new_frame)
        # apply affine transformation to substrate atoms
        substrate_atoms = (
            ms.mint.mm(A, self.motif_substrate_atoms.transpose(0, 1)).transpose(0, 1)
            + t
        )
        second_distance = ms.mint.sqrt(
            ms.mint.sqrt(
                ms.mint.sum(ms.mint.square(new_frame[0] - substrate_atoms[0]), dim=-1)
            )
        )
        assert abs(first_distance - second_distance) < 0.01, "Alignment seems to be bad"
        diffusion_mask = mask_expand(self.diffusion_mask, 1)
        Ca = xyz[~diffusion_mask, 1]

        Ca_exp = Ca[None, :, :]  # [1, Lb, 3]
        subs_exp = substrate_atoms[None, :, :]  # [1, Lb, 3]

        x2 = ms.mint.sum(Ca_exp * Ca_exp, dim=-1, keepdim=True)  # [1, Lb, 1]
        y2 = ms.mint.sum(subs_exp * subs_exp, dim=-1, keepdim=True).swapaxes(
            -2, -1
        )  # [1, 1, Lb]
        cross = ms.mint.matmul(Ca_exp, subs_exp.swapaxes(-2, -1))  # [1, Lb, Lb]

        d2 = ops.relu(x2 + y2 - 2.0 * cross)  # [1, Lb, Lb]
        dgram = ms.mint.sqrt(d2)[0]  # [Lb, Lb]

        all_energies = []
        for i, energy_fn in enumerate(self.energies):
            energy = energy_fn(dgram)
            all_energies.append(energy.sum())
        return -self.weight * sum(all_energies)

    def _recover_affine(self, frame1, frame2):
        """
        Uses Simplex Affine Matrix (SAM) formula to recover affine transform between two sets of 4 xyz coordinates
        See: https://www.researchgate.net/publication/332410209_Beginner%27s_guide_to_mapping_simplexes_affinely

        Args:
        frame1 - 4 coordinates from starting frame [4,3]
        frame2 - 4 coordinates from ending frame [4,3]

        Outputs:
        A - affine transformation matrix from frame1->frame2
        t - affine translation vector from frame1->frame2
        """

        l = len(frame1)
        # construct SAM denominator matrix
        B = ops.vstack([frame1.T, ms.mint.ones(l)])
        D = 1.0 / ops.det(B)  # SAM denominator

        M = ms.mint.zeros((3, 4), dtype=ms.float32)
        for i, R in enumerate(frame2.T):
            for j in range(l):
                num = ops.vstack([R, B])
                # make SAM numerator matrix
                # make numerator matrix
                num = ms.mint.cat((num[: j + 1], num[j + 2 :]))
                # calculate SAM entry
                M[i][j] = (-1) ** j * D * ops.det(num)

        A = M[:, : l - 1]
        t = M[:, l - 1 :]
        t = t.transpose(0, 1)
        return A, t

    def _grab_motif_residues(self, xyz) -> None:
        """
        Grabs 4 atoms in the motif.
        Currently random subset of Ca atoms if the motif is >= 4 residues, or else 4 random atoms from a single residue
        """
        idx = ms.mint.arange(self.diffusion_mask.shape[0])
        idx = idx[self.diffusion_mask].float()
        if ms.mint.sum(self.diffusion_mask) >= 4:
            rand_idx = ms.mint.multinomial(idx, 4).long()
            # get Ca atoms
            self.motif_frame = xyz[rand_idx, 1]
            self.motif_mapping = [(i, 1) for i in rand_idx]
        else:
            rand_idx = ms.mint.multinomial(idx, 1).long()
            self.motif_frame = xyz[rand_idx[0], :4]
            self.motif_mapping = [(rand_idx, i) for i in range(4)]


# Dictionary of types of potentials indexed by name of potential. Used by PotentialManager.
# If you implement a new potential you must add it to this dictionary for it to be used by
# the PotentialManager
implemented_potentials = {
    "monomer_ROG": monomer_ROG,
    "binder_ROG": binder_ROG,
    "dimer_ROG": dimer_ROG,
    "binder_ncontacts": binder_ncontacts,
    "interface_ncontacts": interface_ncontacts,
    "monomer_contacts": monomer_contacts,
    "olig_contacts": olig_contacts,
    "substrate_contacts": substrate_contacts,
}

require_binderlen = {
    "binder_ROG",
    "binder_distance_ReLU",
    "binder_any_ReLU",
    "dimer_ROG",
    "binder_ncontacts",
    "interface_ncontacts",
}
