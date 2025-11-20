#!/usr/bin/env python
"""
Inference script.

To run with base.yaml as the config,

> python run_inference.py

To specify a different config,

> python run_inference.py --config-name symmetry

where symmetry can be the filename of any other config (without .yaml extension)
See https://hydra.cc/docs/advanced/hydra-command-line-flags/ for more options.

"""

import glob
import logging
import os
import pickle
import random
import re
import time

import hydra
import mindspore as ms
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from rfdiffusion.chemical import num2aa
from rfdiffusion.inference import utils as iu
from rfdiffusion.inference.ab_util import ab_write_pdblines
from rfdiffusion.util import generate_Cbeta, writepdb, writepdb_multi


def make_deterministic(seed=0):
    ms.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


@hydra.main(version_base=None, config_path="./config/inference", config_name="base")
def main(conf: HydraConfig) -> None:
    log = logging.getLogger(__name__)
    if conf.inference.deterministic:
        make_deterministic()

    # Initialize sampler and target/contig.
    sampler = iu.sampler_selector(conf)

    # Loop over number of designs to sample.
    design_startnum = sampler.inf_conf.design_startnum
    if sampler.inf_conf.design_startnum == -1:
        existing = glob.glob(sampler.inf_conf.output_prefix + "*.pdb")
        indices = [-1]
        for e in existing:
            print(e)
            m = re.match(".*_(\d+)\.pdb$", e)
            print(m)
            if not m:
                continue
            m = m.groups()[0]
            indices.append(int(m))
        design_startnum = max(indices) + 1

    for i_des in range(design_startnum, design_startnum + sampler.inf_conf.num_designs):
        if conf.inference.deterministic:
            make_deterministic(i_des)

        start_time = time.time()
        out_prefix = f"{sampler.inf_conf.output_prefix}_{i_des}"
        log.info(f"Making design {out_prefix}")
        if sampler.inf_conf.cautious and os.path.exists(out_prefix + ".pdb"):
            log.info(
                f"(cautious mode) Skipping this design because {out_prefix}.pdb already exists."
            )
            continue

        x_init, seq_init = sampler.sample_init()
        denoised_xyz_stack = []
        px0_xyz_stack = []
        seq_stack = []
        plddt_stack = []

        x_t = ms.mint.clone(x_init)
        seq_t = ms.mint.clone(seq_init)

        # Loop over number of reverse diffusion time steps.
        for t in range(int(sampler.t_step_input), sampler.inf_conf.final_step - 1, -1):
            px0, x_t, seq_t, plddt = sampler.sample_step(
                t=t, x_t=x_t, seq_init=seq_t, final_step=sampler.inf_conf.final_step
            )
            px0_xyz_stack.append(px0)
            denoised_xyz_stack.append(x_t)
            seq_stack.append(seq_t)
            plddt_stack.append(plddt[0])  # remove singleton leading dimension

        # Flip order for better visualization in pymol
        denoised_xyz_stack = ms.mint.stack(denoised_xyz_stack)
        denoised_xyz_stack = ms.mint.flip(
            denoised_xyz_stack,
            [
                0,
            ],
        )
        px0_xyz_stack = ms.mint.stack(px0_xyz_stack)
        px0_xyz_stack = ms.mint.flip(
            px0_xyz_stack,
            [
                0,
            ],
        )

        # For logging -- don't flip
        plddt_stack = ms.mint.stack(plddt_stack)

        # Save outputs
        os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
        final_seq = seq_stack[-1]

        # Output glycines, except for motif region
        final_seq = ms.mint.where(
            ms.mint.argmax(seq_init, dim=-1) == 21, 7, ms.mint.argmax(seq_init, dim=-1)
        )  # 7 is glycine

        bfacts = ms.mint.ones_like(final_seq.squeeze())
        # make bfact=0 for diffused coordinates
        bfacts[ms.mint.where(ms.mint.argmax(seq_init, dim=-1) == 21, True, False)] = 0
        # pX0 last step
        out = f"{out_prefix}.pdb"

        # Now don't output sidechains
        if sampler.ab_design():
            # Also mark hotspots in ab design
            bfacts[sampler.ab_item.hotspots] = 0

            # Write as PDB files
            pdblines = ab_write_pdblines(
                atoms=denoised_xyz_stack[0, :, :4].numpy(),
                seq=final_seq.numpy(),
                chain_idx=sampler.chain_idx,
                bfacts=bfacts.numpy(),
                loop_map=sampler.loop_map,
                num2aa=num2aa,
            )

            with open(out, "w", encoding="utf-8") as f_out:
                f_out.write("\n".join(pdblines))

        else:
            writepdb(
                out,
                denoised_xyz_stack[0, :, :4],
                final_seq,
                sampler.binderlen,
                chain_idx=sampler.chain_idx,
                bfacts=bfacts,
                idx_pdb=sampler.idx_pdb,
            )

        # run metadata
        trb = dict(
            config=OmegaConf.to_container(sampler._conf, resolve=True),
            plddt=plddt_stack.numpy(),
            time=time.time() - start_time,
        )

        if sampler.ab_design():
            for loop in sampler.loop_map:
                trb[f"{loop.upper()}_len"] = len(sampler.loop_map[loop])

        if (
            sampler.ab_design()
            and ms.mint.any(sampler.ab_item.target_mask)
            and ms.mint.any(sampler.ab_item.hotspots)
        ):
            # Loop through the hotspots, find the closest loop residue by Cb distance
            # And then average the distance over each of the hotspots. Report min and mean distance
            Cb = generate_Cbeta(
                N=denoised_xyz_stack[0, :, 0],
                Ca=denoised_xyz_stack[0, :, 1],
                C=denoised_xyz_stack[0, :, 2],
            )

            # We are going to assume constructed Cb is identical to original Cb - NRB
            dist = ms.mint.cdist(
                Cb[sampler.ab_item.hotspots], Cb[sampler.ab_item.loop_mask]
            )  # [hotspot_L, loop_L]

            mindist = ms.mint.min(dist, dim=1)[0]  # The min distance for each hotspot

            overallmin = ms.mint.min(
                mindist
            )  # The distance of the closest hotspot to a loop
            averagemin = ms.mint.mean(
                mindist
            )  # The average distance of the hotspots to a loop

            print(f"Overall min distance hotspot to designed loop: {overallmin}")
            print(f"Average min distance hotspot to designed loop: {averagemin}")

            trb["mindist"] = overallmin.numpy()
            trb["averagemin"] = averagemin.numpy()

        if hasattr(sampler, "contig_map"):
            for key, value in sampler.contig_map.get_mappings().items():
                trb[key] = value
        with open(f"{out_prefix}.trb", "wb") as f_out:
            pickle.dump(trb, f_out)

        if sampler.inf_conf.write_trajectory:
            # trajectory pdbs
            traj_prefix = (
                os.path.dirname(out_prefix) + "/traj/" + os.path.basename(out_prefix)
            )
            os.makedirs(os.path.dirname(traj_prefix), exist_ok=True)

            out = f"{traj_prefix}_Xt-1_traj.pdb"
            writepdb_multi(
                out,
                denoised_xyz_stack,
                bfacts,
                final_seq.squeeze(),
                use_hydrogens=False,
                backbone_only=False,
                chain_ids=sampler.chain_idx,
            )

            out = f"{traj_prefix}_pX0_traj.pdb"
            writepdb_multi(
                out,
                px0_xyz_stack,
                bfacts,
                final_seq.squeeze(),
                use_hydrogens=False,
                backbone_only=False,
                chain_ids=sampler.chain_idx,
            )

        log.info(f"Finished design in {(time.time()-start_time)/60:.2f} minutes")


if __name__ == "__main__":
    main()
