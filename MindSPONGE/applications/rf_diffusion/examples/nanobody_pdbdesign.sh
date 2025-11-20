#!/bin/bash

python run_inference.py \
    --config-name antibody \
    antibody.target_pdb=./examples/antibody_pdbs/rsv_site3.pdb \
    antibody.framework_pdb=./examples/antibody_pdbs/h-NbBCII10.pdb \
    inference.ckpt_override_path=./models/RFdiffusion_Ab.ckpt \
    'ppi.hotspot_res=[T305,T456]' \
    'antibody.design_loops=[L1:8-13,L2:7,L3:9-11,H1:7,H2:6,H3:5-13]' \
    inference.num_designs=2 \
    inference.deterministic=True \
    diffuser.T=50 \
    inference.output_prefix=example_outputs/nb_des
