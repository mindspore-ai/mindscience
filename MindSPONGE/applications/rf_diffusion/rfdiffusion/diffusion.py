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


import logging
import math
import os
import pickle

# script for diffusion protocols
import mindspore as ms
import numpy as np
from mindspore import ops
from scipy.spatial.transform import Rotation as scipy_R

from . import igso3
from .util import rigid_from_3_points


def get_beta_schedule(T, b0, bT, schedule_type, inference=False):
    """
    Given a noise schedule type, create the beta schedule
    """
    assert schedule_type in ["linear"]

    # Adjust b0 and bT if T is not 200
    # This is a good approximation, with the beta correction below, unless T is very small
    assert T >= 15, "With discrete time and T < 15, the schedule is badly approximated"
    b0 *= 200 / T
    bT *= 200 / T

    # linear noise schedule
    if schedule_type == "linear":
        schedule = ms.mint.linspace(b0, bT, T)

    else:
        raise NotImplementedError(f"Schedule of type {schedule_type} not implemented.")

    # get alphabar_t for convenience
    alpha_schedule = 1 - schedule
    alphabar_t_schedule = ms.mint.cumprod(alpha_schedule, dim=0)

    if inference:
        print(
            f"With this beta schedule ({schedule_type} schedule, beta_0 = {round(b0, 3)}, beta_T = {round(bT,3)}), alpha_bar_T = {alphabar_t_schedule[-1]}"
        )

    return schedule, alpha_schedule, alphabar_t_schedule


class EuclideanDiffuser:
    # class for diffusing points in 3D

    def __init__(
        self,
        T,
        b_0,
        b_T,
        schedule_type="linear",
        schedule_kwargs=None,
    ):
        self.T = T

        # make noise/beta schedule
        if schedule_kwargs is None:
            schedule_kwargs = {}
        (
            self.beta_schedule,
            self.alpha_schedule,
            self.alphabar_schedule,
        ) = get_beta_schedule(T, b_0, b_T, schedule_type, **schedule_kwargs)

    def diffuse_translations(self, xyz, diffusion_mask=None, var_scale=1):
        return self.apply_kernel_recursive(xyz, diffusion_mask, var_scale)

    def apply_kernel(self, x, t, diffusion_mask=None, var_scale=1):
        """
        Applies a noising kernel to the points in x

        Parameters:
            x (mindspore.tensor, required): (N,3,3) set of backbone coordinates

            t (int, required): Which timestep

            noise_scale (float, required): scale for noise
        """
        t_idx = t - 1  # bring from 1-indexed to 0-indexed

        assert len(x.shape) == 3
        L, _, _ = x.shape

        # c-alpha crds
        ca_xyz = x[:, 1, :]

        b_t = self.beta_schedule[t_idx]

        # get the noise at timestep t
        mean = ms.mint.sqrt(1 - b_t) * ca_xyz
        var = ms.mint.ones((L, 3)) * (b_t) * var_scale

        sampled_crds = ms.mint.normal(mean, ms.mint.sqrt(var))
        delta = sampled_crds - ca_xyz

        if diffusion_mask is not None:
            delta[diffusion_mask, ...] = 0

        out_crds = x + delta[:, None, :]

        return out_crds, delta

    def apply_kernel_recursive(self, xyz, diffusion_mask=None, var_scale=1):
        """
        Repeatedly apply self.apply_kernel T times and return all crds
        """
        bb_stack = []
        T_stack = []

        cur_xyz = ms.mint.clone(xyz)

        for t in range(1, self.T + 1):
            cur_xyz, cur_T = self.apply_kernel(
                cur_xyz, t, var_scale=var_scale, diffusion_mask=diffusion_mask
            )
            bb_stack.append(cur_xyz)
            T_stack.append(cur_T)

        return ms.mint.stack(bb_stack).transpose(0, 1), ms.mint.stack(
            T_stack
        ).transpose(0, 1)


def write_pkl(save_path: str, pkl_data):
    """Serialize data into a pickle file."""
    with open(save_path, "wb") as handle:
        pickle.dump(pkl_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_pkl(read_path: str, verbose=False):
    """Read data from a pickle file."""
    with open(read_path, "rb") as handle:
        try:
            return pickle.load(handle)
        except Exception as e:
            if verbose:
                print(f"Failed to read {read_path}")
            raise e


class IGSO3:
    """
    Class for taking in a set of backbone crds and performing IGSO3 diffusion
    on all of them.

    Unlike the diffusion on translations, much of this class is written for a
    scaling between an initial time t=0 and final time t=1.
    """

    def __init__(
        self,
        *,
        T,
        min_sigma,
        max_sigma,
        min_b,
        max_b,
        cache_dir,
        num_omega=1000,
        schedule="linear",
        L=2000,
    ):
        """

        Args:
            T: total number of time steps
            min_sigma: smallest allowed scale parameter, should be at least 0.01 to maintain numerical stability.  Recommended value is 0.05.
            max_sigma: for exponential schedule, the largest scale parameter. Ignored for recommended linear schedule
            min_b: lower value of beta in Ho schedule analogue
            max_b: upper value of beta in Ho schedule analogue
            num_omega: discretization level in the angles across [0, pi]
            schedule: currently only linear and exponential are supported.  The exponential schedule may be noising too slowly.
            L: truncation level
        """
        self._log = logging.getLogger(__name__)

        self.T = T

        self.schedule = schedule
        self.cache_dir = cache_dir
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

        if self.schedule == "linear":
            self.min_b = min_b
            self.max_b = max_b
            self.max_sigma = self.sigma(1.0)
        self.num_omega = num_omega
        self.num_sigma = 500
        # Calculate igso3 values.
        self.L = L  # truncation level
        self.igso3_vals = self._calc_igso3_vals(L=L)
        self.step_size = 1 / self.T

    def _calc_igso3_vals(self, L=2000):
        """_calc_igso3_vals computes numerical approximations to the
        relevant analytically intractable functionals of the igso3
        distribution.

        The calculated values are cached, or loaded from cache if they already
        exist.

        Args:
            L: truncation level for power series expansion of the pdf.
        """

        def replace_period(x):
            return str(x).replace(".", "_")

        if self.schedule == "linear":
            cache_fname = os.path.join(
                self.cache_dir,
                f"T_{self.T}_omega_{self.num_omega}_min_sigma_{replace_period(self.min_sigma)}"
                + f"_min_b_{replace_period(self.min_b)}_max_b_{replace_period(self.max_b)}_schedule_{self.schedule}.pkl",
            )
        elif self.schedule == "exponential":
            cache_fname = os.path.join(
                self.cache_dir,
                f"T_{self.T}_omega_{self.num_omega}_min_sigma_{replace_period(self.min_sigma)}"
                f"_max_sigma_{replace_period(self.max_sigma)}_schedule_{self.schedule}",
            )
        else:
            raise ValueError(f"Unrecognize schedule {self.schedule}")

        if not os.path.isdir(self.cache_dir):
            os.makedirs(self.cache_dir)

        if os.path.exists(cache_fname):
            self._log.info("Using cached IGSO3.")
            igso3_vals = read_pkl(cache_fname)
        else:
            self._log.info("Calculating IGSO3.")
            igso3_vals = igso3.calculate_igso3(
                num_sigma=self.num_sigma,
                min_sigma=self.min_sigma,
                max_sigma=self.max_sigma,
                num_omega=self.num_omega,
            )
            write_pkl(cache_fname, igso3_vals)

        return igso3_vals

    @property
    def discrete_sigma(self):
        return self.igso3_vals["discrete_sigma"]

    def sigma_idx(self, sigma: np.ndarray):
        """
        Calculates the index for discretized sigma during IGSO(3) initialization."""
        return np.digitize(sigma, self.discrete_sigma) - 1

    def t_to_idx(self, t: np.ndarray):
        """
        Helper function to go from discrete time index t to corresponding sigma_idx.

        Args:
            t: time index (integer between 1 and 200)
        """
        continuous_t = t / self.T
        return self.sigma_idx(self.sigma(continuous_t))

    def sigma(self, t: ms.Tensor):
        """
        Extract \sigma(t) corresponding to chosen sigma schedule.

        Args:
            t: mindspore tensor with time between 0 and 1
        """
        if not type(t) == ms.Tensor:
            t = ms.tensor(t)
        if ms.mint.any(t < 0) or ms.mint.any(t > 1):
            raise ValueError(f"Invalid t={t}")
        if self.schedule == "exponential":
            sigma = t * np.log10(self.max_sigma) + (1 - t) * np.log10(self.min_sigma)
            return 10**sigma
        if self.schedule == "linear":  # Variance exploding analogue of Ho schedule
            # add self.min_sigma for stability
            return (
                self.min_sigma
                + t * self.min_b
                + (1 / 2) * (t**2) * (self.max_b - self.min_b)
            )

        raise ValueError(f"Unrecognize schedule {self.schedule}")

    def g(self, t):
        """
        g returns the drift coefficient at time t

        since
            sigma(t)^2 := \int_0^t g(s)^2 ds,
        for arbitrary sigma(t) we invert this relationship to compute
            g(t) = sqrt(d/dt sigma(t)^2).

        Args:
            t: scalar time between 0 and 1

        Returns:
            drift cooeficient as a scalar.
        """
        t = ms.tensor(t)

        def sigma_sqr(t):
            return (self.sigma(t) ** 2).sum()

        grads = ops.grad(sigma_sqr)(t)
        return ms.mint.sqrt(grads)

    def sample(self, ts, n_samples=1):
        """
        sample uses the inverse cdf to sample an angle of rotation from
        IGSO(3)

        Args:
            ts: array of integer time steps to sample from.
            n_samples: number of samples to draw.
        Returns:
        sampled angles of rotation. [len(ts), N]
        """
        assert sum(ts == 0) == 0, "assumes one-indexed, not zero indexed"
        all_samples = []
        for t in ts:
            sigma_idx = self.t_to_idx(t)
            sample_i = np.interp(
                np.random.rand(n_samples),
                self.igso3_vals["cdf"][sigma_idx],
                self.igso3_vals["discrete_omega"],
            )  # [N, 1]
            all_samples.append(sample_i)
        return np.stack(all_samples, axis=0)

    def sample_vec(self, ts, n_samples=1):
        """sample_vec generates a rotation vector(s) from IGSO(3) at time steps
        ts.

        Return:
            Sampled vector of shape [len(ts), N, 3]
        """
        x = np.random.randn(len(ts), n_samples, 3)
        x /= np.linalg.norm(x, axis=-1, keepdims=True)
        return x * self.sample(ts, n_samples=n_samples)[..., None]

    def score_norm(self, t, omega):
        """
        score_norm computes the score norm based on the time step and angle
        Args:
            t: integer time step
            omega: angles (scalar or shape [N])
        Return:
            score_norm with same shape as omega
        """
        sigma_idx = self.t_to_idx(t)
        score_norm_t = np.interp(
            omega,
            self.igso3_vals["discrete_omega"],
            self.igso3_vals["score_norm"][sigma_idx],
        )
        return score_norm_t

    def score_vec(self, ts, vec):
        """score_vec computes the score of the IGSO(3) density as a rotation
        vector. This score vector is in the direction of the sampled vector,
        and has magnitude given by score_norms.

        In particular, Rt @ hat(score_vec(ts, vec)) is what is referred to as
        the score approximation in Algorithm 1


        Args:
            ts: times of shape [T]
            vec: where to compute the score of shape [T, N, 3]
        Returns:
            score vectors of shape [T, N, 3]
        """
        omega = np.linalg.norm(vec, axis=-1)
        all_score_norm = []
        for i, t in enumerate(ts):
            omega_t = omega[i]
            t_idx = t - 1
            sigma_idx = self.t_to_idx(t)
            score_norm_t = np.interp(
                omega_t,
                self.igso3_vals["discrete_omega"],
                self.igso3_vals["score_norm"][sigma_idx],
            )[:, None]
            all_score_norm.append(score_norm_t)
        score_norm = np.stack(all_score_norm, axis=0)
        return score_norm * vec / omega[..., None]

    def exp_score_norm(self, ts):
        """exp_score_norm returns the expected value of norm of the score for
        IGSO(3) with time parameter ts of shape [T].
        """
        sigma_idcs = [self.t_to_idx(t) for t in ts]
        return self.igso3_vals["exp_score_norms"][sigma_idcs]

    def diffuse_frames(self, xyz, t_list, diffusion_mask=None):
        """diffuse_frames samples from the IGSO(3) distribution to noise frames

        Parameters:
            xyz (np.array or mindspore.tensor, required): (L,3,3) set of backbone coordinates
            mask (np.array or mindspore.tensor, required): (L,) set of bools. True/1 is NOT diffused, False/0 IS diffused
        Returns:
            np.array : N/CA/C coordinates for each residue
                        (T,L,3,3), where T is num timesteps
        """

        if ms.is_tensor(xyz):
            xyz = xyz.numpy()

        t = np.arange(self.T) + 1  # 1-indexed!!
        num_res = len(xyz)

        N = ms.from_numpy(xyz[None, :, 0, :])
        Ca = ms.from_numpy(xyz[None, :, 1, :])  # [1, num_res, 3, 3]
        C = ms.from_numpy(xyz[None, :, 2, :])

        # scipy rotation object for true coordinates
        R_true, Ca = rigid_from_3_points(N, Ca, C)
        R_true = R_true[0]
        Ca = Ca[0]

        # Sample rotations and scores from IGSO3
        sampled_rots = self.sample_vec(t, n_samples=num_res)  # [T, N, 3]

        if diffusion_mask is not None:
            non_diffusion_mask = 1 - diffusion_mask[None, :, None]
            sampled_rots = sampled_rots * non_diffusion_mask

        # Apply sampled rot.
        R_sampled = (
            scipy_R.from_rotvec(sampled_rots.reshape(-1, 3))
            .as_matrix()
            .reshape(self.T, num_res, 3, 3)
        )
        R_perturbed = np.einsum("tnij,njk->tnik", R_sampled, R_true)
        perturbed_crds = (
            np.einsum(
                "tnij,naj->tnai", R_sampled, xyz[:, :3, :] - Ca[:, None, ...].numpy()
            )
            + Ca[None, :, None].numpy()
        )

        if t_list is not None:
            idx = [i - 1 for i in t_list]
            perturbed_crds = perturbed_crds[idx]
            R_perturbed = R_perturbed[idx]

        return (
            perturbed_crds.transpose(1, 0, 2, 3),  # [L, T, 3, 3]
            R_perturbed.transpose(1, 0, 2, 3),
        )

    def reverse_sample_vectorized(
        self, R_t, R_0, t, noise_level, mask=None, return_perturb=False
    ):
        """reverse_sample uses an approximation to the IGSO3 score to sample
        a rotation at the previous time step.

        Roughly - this update follows the reverse time SDE for Reimannian
        manifolds proposed by de Bortoli et al. Theorem 1 [1]. But with an
        approximation to the score based on the prediction of R0.
        Unlike in reference [1], this diffusion on SO(3) relies on geometric
        variance schedule.  Specifically we follow [2] (appendix C) and assume
            sigma_t = sigma_min * (sigma_max / sigma_min)^{t/T},
        for time step t.  When we view this as a discretization  of the SDE
        from time 0 to 1 with step size (1/T).  Following Eq. 5 and Eq. 6,
        this maps on to the forward  time SDEs
            dx = g(t) dBt [FORWARD]
        and
            dx = g(t)^2 score(xt, t)dt + g(t) B't, [REVERSE]
        where g(t) = sigma_t * sqrt(2 * log(sigma_max/ sigma_min)), and Bt and
        B't are Brownian motions. The formula for g(t) obtains from equation 9
        of [2], from which this sampling function may be generalized to
        alternative noising schedules.
        Args:
            R_t: noisy rotation of shape [N, 3, 3]
            R_0: prediction of un-noised rotation
            t: integer time step
            noise_level: scaling on the noise added when obtaining sample
                (preliminary performance seems empirically better with noise
                level=0.5)
            mask: whether the residue is to be updated.  A value of 1 means the
                rotation is not updated from r_t.  A value of 0 means the
                rotation is updated.
        Return:
            sampled rotation matrix for time t-1 of shape [3, 3]
        Reference:
        [1] De Bortoli, V., Mathieu, E., Hutchinson, M., Thornton, J., Teh, Y.
        W., & Doucet, A. (2022). Riemannian score-based generative modeling.
        arXiv preprint arXiv:2202.02763.
        [2] Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S.,
        & Poole, B. (2020). Score-based generative modeling through stochastic
        differential equations. arXiv preprint arXiv:2011.13456.
        """
        # compute rotation vector corresponding to prediction of how r_t goes to r_0
        R_0, R_t = ms.tensor(R_0), ms.tensor(R_t)
        R_0t = ms.mint.einsum("...ij,...kj->...ik", R_t, R_0)
        R_0t_rotvec = ms.tensor(scipy_R.from_matrix(R_0t.numpy()).as_rotvec())

        # Approximate the score based on the prediction of R0.
        # R_t @ hat(Score_approx) is the score approximation in the Lie algebra
        # SO(3) (i.e. the output of Algorithm 1)
        Omega = ops.norm(R_0t_rotvec, dim=-1).numpy()
        Score_approx = (
            R_0t_rotvec * ms.from_numpy(self.score_norm(t, Omega) / Omega)[:, None]
        )

        # Compute scaling for score and sampled noise (following Eq 6 of [2])
        continuous_t = t / self.T
        rot_g = self.g(continuous_t)

        # Sample and scale noise to add to the rotation perturbation in the
        # SO(3) tangent space.  Since IG-SO(3) is the Brownian motion on SO(3)
        # (up to a deceleration of time by a factor of two), for small enough
        # time-steps, this is equivalent to perturbing r_t with IG-SO(3) noise.
        # See e.g. Algorithm 1 of De Bortoli et al.
        Z = np.random.normal(size=(R_0.shape[0], 3))
        Z = ms.from_numpy(Z)
        Z *= noise_level

        Delta_r = (rot_g**2) * self.step_size * Score_approx

        # Sample perturbation from discretized SDE (following eq. 6 of [2]),
        # This approximate sampling from IGSO3(* ; Delta_r, rot_g^2 *
        # self.step_size) with tangent Gaussian.
        Perturb_tangent = Delta_r + rot_g * math.sqrt(self.step_size) * Z
        if mask is not None:
            Perturb_tangent *= (1 - mask.long())[:, None, None]
        Perturb = igso3.Exp(Perturb_tangent)

        if return_perturb:
            return Perturb

        Interp_rot = ms.mint.einsum("...ij,...jk->...ik", Perturb, R_t)

        return Interp_rot


class Diffuser:
    # wrapper for yielding diffused coordinates

    def __init__(
        self,
        T,
        b_0,
        b_T,
        min_sigma,
        max_sigma,
        min_b,
        max_b,
        schedule_type,
        so3_schedule_type,
        so3_type,
        crd_scale,
        schedule_kwargs=None,
        var_scale=1.0,
        cache_dir=".",
        partial_T=None,
        truncation_level=2000,
    ):
        """
        Parameters:

            T (int, required): Number of steps in the schedule

            b_0 (float, required): Starting variance for Euclidean schedule

            b_T (float, required): Ending variance for Euclidean schedule

        """
        self.T = T
        self.b_0 = b_0
        self.b_T = b_T
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.crd_scale = crd_scale
        self.var_scale = var_scale
        self.cache_dir = cache_dir

        if schedule_kwargs is None:
            schedule_kwargs = {}

        # get backbone frame diffuser
        self.so3_diffuser = IGSO3(
            T=self.T,
            min_sigma=self.min_sigma,
            max_sigma=self.max_sigma,
            schedule=so3_schedule_type,
            min_b=min_b,
            max_b=max_b,
            cache_dir=self.cache_dir,
            L=truncation_level,
        )

        # get backbone translation diffuser
        self.eucl_diffuser = EuclideanDiffuser(
            self.T, b_0, b_T, schedule_type=schedule_type, **schedule_kwargs
        )

        print("Successful diffuser __init__")

    def diffuse_pose(
        self,
        xyz,
        seq,
        atom_mask,
        include_motif_sidechains=True,
        diffusion_mask=None,
        t_list=None,
    ):
        """
        Given full atom xyz, sequence and atom mask, diffuse the protein frame
        translations and rotations

        Parameters:

            xyz (L,14/27,3) set of coordinates

            seq (L,) integer sequence

            atom_mask: mask describing presence/absence of an atom in pdb

            diffusion_mask (mindspore.tensor, optional): Tensor of bools, True means NOT diffused at this residue, False means diffused

            t_list (list, optional): If present, only return the diffused coordinates at timesteps t within the list


        """

        if diffusion_mask is None:
            diffusion_mask = ms.mint.zeros(len(xyz.squeeze())).to(dtype=ms.bool_)

        L = len(xyz)

        # bring to origin and scale
        # check if any BB atoms are nan before centering
        nan_mask = ~ms.mint.isnan(xyz.squeeze()[:, :3]).any(dim=-1).any(dim=-1)
        assert ms.mint.sum(~nan_mask) == 0

        # Centre unmasked structure at origin, as in training (to prevent information leak)
        if ms.mint.sum(diffusion_mask) != 0:
            self.motif_com = xyz[diffusion_mask, 1, :].mean(
                dim=0
            )  # This is needed for one of the potentials
            xyz = xyz - self.motif_com
        elif ms.mint.sum(diffusion_mask) == 0:
            xyz = xyz - xyz[:, 1, :].mean(dim=0)

        xyz_true = ms.mint.clone(xyz)
        xyz = xyz * self.crd_scale

        # 1 get translations
        diffused_T, deltas = self.eucl_diffuser.diffuse_translations(
            xyz[:, :3, :].clone(), diffusion_mask=diffusion_mask
        )
        # print('Time to diffuse coordinates: ',time.time()-tick)
        diffused_T /= self.crd_scale
        deltas /= self.crd_scale

        # 2 get frames
        diffused_frame_crds, diffused_frames = self.so3_diffuser.diffuse_frames(
            xyz[:, :3, :].clone(), diffusion_mask=diffusion_mask.numpy(), t_list=None
        )
        diffused_frame_crds /= self.crd_scale
        # print('Time to diffuse frames: ',time.time()-tick)

        # Now combine all the diffused quantities to make full atom diffused poses
        cum_delta = deltas.cumsum(dim=1)
        # The coordinates of the translated AND rotated frames
        diffused_BB = (
            ms.from_numpy(diffused_frame_crds) + cum_delta[:, :, None, :]
        ).transpose(
            0, 1
        )  # [n,L,3,3]

        # diffused_BB is [t_steps,L,3,3]
        t_steps, L = diffused_BB.shape[:2]

        diffused_fa = ms.mint.zeros((t_steps, L, 27, 3))
        diffused_fa[:, :, :3, :] = diffused_BB

        # Add in sidechains from motif
        if include_motif_sidechains:
            diffused_fa[:, diffusion_mask, :14, :] = xyz_true[None, diffusion_mask, :14]

        if t_list is None:
            fa_stack = diffused_fa
        else:
            t_idx_list = [t - 1 for t in t_list]
            fa_stack = diffused_fa[t_idx_list]

        return fa_stack, xyz_true
