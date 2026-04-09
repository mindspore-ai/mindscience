---
name: rfdiffusion
description: RFdiffusion is a deep learning method for protein structure generation using diffusion models. Use this model when you need to design novel protein structures, including unconditional protein generation, motif scaffolding, binder design, symmetric oligomer design, antibody/nanobody design, and macrocyclic peptide design.
license: BSD License (original), Apache License 2.0 (MindSpore implementation)
metadata:
    skill-author: MindSpore Science Team
---

# RFdiffusion

**IMPORTANT:** This SKILL.md should only document **inference-related** functionality. Do not include training, fine-tuning, evaluation, or other non-inference scenarios. All sections should focus on how to use the model for inference/prediction tasks only.

## Overview

RFdiffusion is an open-source protein structure generation method based on diffusion models that can run with or without conditioning information (a motif, target, antibody frameworks, etc.). This implementation is based on MindSpore and adapted from the original [RFdiffusion](https://github.com/RosettaCommons/RFdiffusion) repository, with integration for [RFantibody](https://github.com/RosettaCommons/RFantibody) antibody design modules.

Key capabilities:
- Unconditional protein generation (de novo protein design)
- Motif scaffolding (scaffold functional motifs with new protein structures)
- Symmetric oligomer generation (cyclic, dihedral, tetrahedral symmetries)
- Binder design (design proteins that bind to target structures)
- Antibody/nanobody design (RFantibody integration for CDR loop design)
- Macrocyclic peptide design (RFpeptides protocol)
- Partial diffusion (design diversification around existing structures)

---

## When to Use

### Primary Application Scenarios

| Scenario | Description | Example Use Case |
|----------|-------------|------------------|
| **Unconditional Protein Generation** | Generate novel protein structures from scratch without any constraints | Designing new protein scaffolds for engineering applications |
| **Motif Scaffolding** | Scaffold functional motifs (e.g., enzyme active sites, binding sites) with new protein backbones | Creating scaffolds for enzyme active sites or binding motifs |
| **Binder Design** | Design protein binders to target structures with specified hotspot residues | Designing therapeutic binders to cell-surface receptors |
| **Antibody/Nanobody Design** | Design antibody Fv or nanobody VHH interfaces with CDR loop redesign | Developing antibody therapeutics with optimized interfaces |

### Secondary Application Scenarios

| Scenario | Description | Example Use Case |
|----------|-------------|------------------|
| **Symmetric Oligomer Design** | Generate symmetric protein assemblies (cyclic, dihedral, tetrahedral) | Designing symmetric nanomaterials or self-assembling proteins |
| **Macrocyclic Peptide Design** | Design cyclic peptides that bind target proteins with atomic accuracy | Developing peptide therapeutics with improved stability |
| **Partial Diffusion** | Diversify existing designs by partial noise/denoise process | Exploring design variations around successful candidates |
| **Fold-Conditioned Design** | Design proteins conditioned on specific topologies/folds | Generating proteins with desired secondary structure arrangements |

---

## How It Works

### 1. Dataset Acquisition and Processing

#### Input Data Requirements

| Requirement | Description |
|-------------|-------------|
| **Data Format** | PDB files (standard protein structure format) for input structures |
| **Input Types** | Target PDB (for binder design), Motif PDB (for scaffolding), Framework PDB (for antibody design) |
| **Data Size** | Single PDB files; batch processing via `inference.num_designs` parameter |
| **Data Source** | Experimental structures (PDB database), predicted structures (AlphaFold, RoseTTAFold), or custom structures |

#### Data Acquisition Methods

1. **PDB Database**: Download structures from RCSB PDB (https://www.rcsb.org/) - Standard source for experimental structures
2. **Structure Prediction**: Use AlphaFold, RoseTTAFold, or MEGA-Protein - For predicted target structures
3. **Example Data**: Use provided example inputs - Quick testing and validation

```bash
# Download example data after installation
tar -xvf examples/ppi_scaffolds_subset.tar.gz -C examples/
tar -xvf examples/antibody_pdbs.tar.gz -C examples/
tar -xvf examples/input_pdbs.tar.gz -C examples/
tar -xvf examples/target_folds.tar.gz -C examples/
tar -xvf examples/tim_barrel_scaffold.tar.gz -C examples/
```

#### Data Preprocessing

- **PDB Preparation**: Ensure PDB files have correct chain IDs and residue numbering
- **Symmetric Motif Scaffolding**: For symmetric scaffolding, input PDB must contain symmetrized motif aligned to canonical symmetry axes
- **Fold Conditioning**: Generate secondary structure (.ms) and block adjacency files using helper scripts

#### Contig Map Specification

The contig map defines how input motifs are connected and how many residues to generate:

| Syntax | Meaning | Example |
|--------|---------|---------|
| `[150-150]` | Generate exactly 150 residues | Unconditional monomer of length 150 |
| `[100-200]` | Generate 100-200 residues (randomly sampled) | Length-flexible design |
| `[10-40/A163-181/10-40]` | Scaffold motif A163-181 with 10-40 residues on each side | Motif scaffolding |
| `[A1-150/0 70-100]` | Target chain A1-150, chain break, generate 70-100 residue binder | Binder design |
| `/0` | Chain break indicator | Separate chains in output |

---

### 2. Environment Configuration and Dependencies

#### Dependency Installation

```bash
# Clone the repository
git clone https://gitee.com/mindspore/mindscience.git
cd mindscience/MindSPONGE/applications/rf_diffusion

# Install dependencies
pip install -r requirements.txt

# Download model weights
bash scripts/download_models.sh

# Download sharker library (required for SE3 transformer)
git clone https://gitee.com/sunhaoneng/gnn.git
cp -r gnn/sharker env/
rm -r -f gnn

# Set PYTHONPATH before each run
export PYTHONPATH=$PYTHONPATH:$(pwd)/env
```

#### Environment Requirements

| Requirement | Specification |
|-------------|---------------|
| **Python Version** | Python >= 3.11 |
| **Framework** | MindSpore >= 2.7.0 |
| **CANN** | CANN >= 8.2.RC1 |
| **MindScience** | mindscience >= 0.8.0 |
| **Hardware** | Ascend NPU (recommended) |
| **Disk Space** | ~2GB for model weights and examples |

#### Required Python Packages

| Package | Version | Purpose |
|---------|---------|---------|
| mindspore | 2.7.1 | Deep learning framework |
| hydra-core | - | Configuration management |
| omegaconf | - | Configuration handling |
| wandb | 0.12.0 | Experiment tracking |
| decorator | 5.1.0 | Utility functions |
| fsspec | - | File system utilities |

#### Available Model Checkpoints

| Checkpoint | Purpose | Use Case |
|------------|---------|----------|
| `Base_ckpt.ckpt` | Base diffusion model | General protein generation |
| `ActiveSite_ckpt.ckpt` | Active site scaffolding | Small motif scaffolding (enzyme active sites) |
| `Complex_base_ckpt.ckpt` | Complex/binder design | PPI binder design |
| `Complex_beta_ckpt.ckpt` | Diverse topology binders | Non-helical binder design |
| `InpaintSeq_ckpt.ckpt` | Sequence inpainting | Motif scaffolding with sequence masking |
| `InpaintSeq_Fold_ckpt.ckpt` | Fold-conditioned inpainting | Fold-conditioned design |
| `Complex_Fold_base_ckpt.ckpt` | Fold-conditioned complex | Scaffold-conditioned binder design |
| `RFdiffusion_Ab.ckpt` | Antibody/nanobody design | RFantibody interface design |

---

### 3. Model Invocation Guide

#### Basic Inference Command Structure

```bash
python run_inference.py \
    [config_options] \
    inference.output_prefix=<output_path> \
    inference.num_designs=<number> \
    'contigmap.contigs=[<contig_specification>]' \
    [additional_options]
```

#### Example 1: Unconditional Monomer Generation

```bash
# Generate 10 unconditional proteins of length 150
python run_inference.py \
    'contigmap.contigs=[150-150]' \
    inference.output_prefix=test_outputs/test \
    inference.num_designs=10

# Generate proteins with flexible length (100-200 residues)
python run_inference.py \
    inference.output_prefix=example_outputs/design_unconditional \
    'contigmap.contigs=[100-200]' \
    inference.num_designs=10
```

#### Example 2: Motif Scaffolding

```bash
# Scaffold motif from input PDB
python run_inference.py \
    inference.output_prefix=example_outputs/design_motifscaffolding \
    inference.input_pdb=examples/input_pdbs/5TPN.pdb \
    'contigmap.contigs=[10-40/A163-181/10-40]' \
    inference.num_designs=10

# For small motifs, use ActiveSite model
python run_inference.py \
    inference.input_pdb=<motif_pdb> \
    'contigmap.contigs=[<contig>]' \
    inference.ckpt_override_path=models/ActiveSite_ckpt.ckpt \
    inference.num_designs=10
```

#### Example 3: Binder Design (PPI)

```bash
# Design binders to target protein with hotspot residues
python run_inference.py \
    inference.output_prefix=example_outputs/design_ppi \
    inference.input_pdb=examples/input_pdbs/insulin_target.pdb \
    'contigmap.contigs=[A1-150/0 70-100]' \
    'ppi.hotspot_res=[A59,A83,A91]' \
    inference.num_designs=10 \
    denoiser.noise_scale_ca=0 \
    denoiser.noise_scale_frame=0
```

#### Example 4: Antibody/Nanobody Design

```bash
# Antibody Fv design
python run_inference.py \
    --config-name antibody \
    antibody.target_pdb=./examples/antibody_pdbs/rsv_site3.pdb \
    antibody.framework_pdb=./examples/antibody_pdbs/hu-4D5-8_Fv.pdb \
    inference.ckpt_override_path=./models/RFdiffusion_Ab.ckpt \
    'ppi.hotspot_res=[T305,T456]' \
    'antibody.design_loops=[L1:8-13,L2:7,L3:9-11,H1:7,H2:6,H3:5-13]' \
    inference.num_designs=2 \
    diffuser.T=50 \
    inference.deterministic=True \
    inference.output_prefix=example_outputs/ab_des

# Nanobody (VHH) design
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
```

#### Example 5: Symmetric Oligomer Design

```bash
# C6 cyclic symmetric oligomer
python run_inference.py \
    --config-name=symmetry \
    inference.symmetry="C6" \
    inference.num_designs=10 \
    inference.output_prefix="example_outputs/C6_oligo" \
    'potentials.guiding_potentials=["type:olig_contacts,weight_intra:1,weight_inter:0.1"]' \
    potentials.olig_intra_all=True \
    potentials.olig_inter_all=True \
    potentials.guide_scale=2.0 \
    potentials.guide_decay="quadratic" \
    'contigmap.contigs=[480-480]'

# Tetrahedral symmetric oligomer
python run_inference.py \
    --config-name=symmetry \
    inference.symmetry="tetrahedral" \
    inference.num_designs=10 \
    inference.output_prefix="example_outputs/tetrahedral_oligo" \
    'potentials.guiding_potentials=["type:olig_contacts,weight_intra:1,weight_inter:0.1"]' \
    potentials.olig_intra_all=True \
    potentials.olig_inter_all=True \
    potentials.guide_scale=2.0 \
    potentials.guide_decay="quadratic" \
    'contigmap.contigs=[1200-1200]'
```

#### Example 6: Macrocyclic Peptide Design

```bash
# Macrocyclic binder design
python run_inference.py \
    --config-name base \
    inference.output_prefix=./outputs/diffused_binder_cyclic2 \
    inference.num_designs=10 \
    'contigmap.contigs=[12-18 A3-117/0]' \
    inference.input_pdb=examples/input_pdbs/7zkr_GABARAP.pdb \
    inference.cyclic=True \
    diffuser.T=50 \
    inference.cyc_chains='a' \
    'ppi.hotspot_res=[A51,A52,A50,A48,A62,A65]'

# Macrocyclic monomer design
python run_inference.py \
    --config-name base \
    inference.output_prefix=./outputs/uncond_cycpep \
    inference.num_designs=10 \
    'contigmap.contigs=[12-18]' \
    inference.cyclic=True \
    diffuser.T=50 \
    inference.cyc_chains='a'
```

#### Example 7: Partial Diffusion

```bash
# Diversify existing design
python run_inference.py \
    inference.input_pdb=<existing_design.pdb> \
    'contigmap.contigs=[100-100/0 B1-150]' \
    diffuser.partial_T=20 \
    inference.num_designs=10
```

---

### 4. Key Configuration Parameters

#### Inference Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `inference.output_prefix` | `samples/design` | Output file path prefix |
| `inference.num_designs` | 10 | Number of designs to generate |
| `inference.input_pdb` | null | Input PDB file path |
| `inference.ckpt_override_path` | null | Override checkpoint path |
| `inference.deterministic` | False | Use deterministic sampling |
| `inference.cyclic` | False | Enable macrocyclic design |
| `inference.cyc_chains` | 'a' | Chains to cyclize |

#### Diffusion Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `diffuser.T` | 50 | Number of diffusion timesteps |
| `diffuser.partial_T` | null | Partial diffusion timestep |
| `diffuser.schedule_type` | linear | Noise schedule type |

#### Contig Map Parameters

| Parameter | Description |
|-----------|-------------|
| `contigmap.contigs` | Contig specification string |
| `contigmap.length` | Fixed total length constraint |
| `contigmap.inpaint_seq` | Residues to mask sequence |
| `contigmap.inpaint_str` | Residues to mask structure |
| `contigmap.provide_seq` | Sequence ranges to keep fixed |

#### PPI Parameters

| Parameter | Description |
|-----------|-------------|
| `ppi.hotspot_res` | Target hotspot residues for binder design |

#### Potential Parameters

| Parameter | Description |
|-----------|-------------|
| `potentials.guiding_potentials` | List of guiding potentials |
| `potentials.guide_scale` | Potential strength scale |
| `potentials.guide_decay` | Decay type (constant, linear, quadratic, cubic) |

---

### 5. Output Files

RFdiffusion generates the following output files:

| File | Description |
|------|-------------|
| `.pdb` | Final predicted structure (designed residues as glycine) |
| `.trb` | Metadata including contig used, config, and mapping information |
| `traj/` | Trajectory files (multi-step PDBs for visualization) |

#### TRB File Contents

- `con_ref_pdb_idx` / `con_hal_pdb_idx`: Input-to-output residue mapping
- `con_ref_idx0` / `con_hal_idx0`: 0-indexed mapping without chain info
- `inpaint_seq`: Masked residue details
- Full inference configuration

---

### 6. Usage Limitations and Notes

#### Functional Limitations

| Limitation | Description |
|------------|-------------|
| **Sidechain Prediction** | Designed residues output as glycine; sidechains not predicted reliably |
| **Small Motif Stability** | Base model may not hold very small motifs fixed; use ActiveSite model |
| **Symmetric Input Requirement** | Symmetric motif scaffolding requires symmetrized input PDB aligned to canonical axes |

#### Performance Considerations

| Consideration | Recommendation |
|---------------|----------------|
| **First Run** | IGSO3 calculation takes time on first run; cached for subsequent runs |
| **Timestep Selection** | 20-50 timesteps sufficient for most tasks; 200 steps for maximum quality |
| **Target Size** | Crop large targets around interface for faster binder design |
| **Noise Scale** | Reduce noise (0.5 or 0) for PPI to improve design quality |

#### Scale Limitations

| Limitation | Description |
|------------|-------------|
| **Protein Length** | Practical limit ~200-300 residues for monomers; longer for oligomers |
| **Symmetric Oligomers** | Total length must be divisible by number of chains |
| **Memory** | Large designs may require significant memory; adjust batch processing |

#### Best Practices

1. **Start with no potentials** as baseline, then gradually increase strength
2. **Use hotspot residues** for PPI to guide interface location
3. **Reduce noise scale** for PPI designs (recommended: 0.5 or 0)
4. **Use ActiveSite model** for scaffolding very small motifs
5. **Sample different partial_T values** for partial diffusion tasks

---

### 7. Reference Resources

#### Papers

- [RFdiffusion Paper](https://www.biorxiv.org/content/10.1101/2022.12.09.519842v1) - Main method paper
- [RFantibody Paper](https://www.biorxiv.org/content/10.1101/2024.03.14.585103v2) - Antibody design extension
- [RFpeptides Paper](https://www.biorxiv.org/content/10.1101/2024.07.16.603789v1) - Macrocyclic peptide design

#### Code Repositories

- [Original RFdiffusion](https://github.com/RosettaCommons/RFdiffusion) - PyTorch implementation
- [RFantibody](https://github.com/RosettaCommons/RFantibody) - Antibody design tools
- [MindScience](https://gitee.com/mindspore/mindscience) - MindSpore implementation

#### Documentation

- MindSpore Documentation: https://www.mindspore.cn/
- MindSPONGE Applications: https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/applications

---

## License

> Modified from [RFdiffusion](https://github.com/RosettaCommons/RFdiffusion)  
> Original license: BSD License  
> MindSpore implementation: Apache License 2.0