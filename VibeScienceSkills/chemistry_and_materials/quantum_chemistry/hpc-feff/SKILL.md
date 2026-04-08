---
name: hpc-feff
description: FEFF 10 X-ray absorption spectroscopy (XAS) simulation for EXAFS, XANES, NEXAFS, and DOS calculations. Used for material characterization, local structure determination, and electronic property analysis on HPC clusters.
---

# HPC-FEFF Skill

FEFF 10 is an ab initio calculation program for X-ray absorption spectroscopy (XAS), implementing finite difference method and multiple scattering theory to compute spectra from first principles.

## Scientific Applications

| Application | Use Case |
|------------|----------|
| **EXAFS Analysis** | Determine coordination numbers, bond lengths, and disorder parameters |
| **XANES/NEXAFS** | Probe electronic structure, oxidation states, and site symmetry |
| **DOS Calculation** | Analyze electronic density of states and band structure |
| **Multiple Scattering** | Model complex paths in disordered systems |
| **Temperature Effects** | Simulate thermal disorder using Debye model |

### Absorption Edges

| Edge | Orbital | Typical Elements |
|------|---------|------------------|
| K | 1s | O, N, C (light elements) |
| L | 2s, 2p | Fe, Cu, Zn (transition metals) |
| M | 3s, 3p | Lanthanides, actinides (heavy elements) |

## Quick Start

```
references/01-input-file-structure.md  → Understand FEFF input file anatomy
references/02-calculation-types.md      → Choose calculation mode (EXAFS/XANES/NEXAFS/DOS)
references/03-structure-input.md        → Prepare atomic coordinates and cluster radius
references/04-advanced-features.md      → Configure DEBYE, SPIN, EXCHANGE, MPI
references/error-recovery.md             → Diagnose SCF, cluster, and path errors
```

## Skill Map

```
                    FEFF WORKFLOW
    ┌──────────────────────────────────────────────────┐
    │  01-Input-File-Structure (Anatomy)               │
    │  ─────────────────────────────────────────────  │
    │  02-Calculation-Types    │  04-Advanced-Features │
    │  ───────────────────────│─────────────────────── │
    │  03-Structure-Input                                 │
    │  ──────────────────────────────────────────────── │
    │  05-Error-Recovery                                  │
    └───────────────────────────────────────────────────┘
```

## Key Decision Points

| Question | Guide | Summary |
|----------|-------|---------|
| Calculation type? | `02-calculation-types.md` | EXAFS, XANES, NEXAFS, or DOS |
| Absorption edge? | `01-input-file-structure.md` | K, L, or M edge based on element |
| Cluster radius? | `03-structure-input.md` | RMAX affects calculation range |
| Atomic coordinates? | `03-structure-input.md` | CIF conversion, Angstrom/Bohr units |
| Temperature effects? | `04-advanced-features.md` | DEBYE keyword |
| SCF convergence? | `error-recovery.md` | Adjust SCF parameters |
| Memory issues? | `error-recovery.md` | Reduce cluster size |

## Input File Structure

FEFF calculations are controlled by `feff.inp` using keywords. See [references/01-input-file-structure.md](references/01-input-file-structure.md) for:

- Keyword reference (EDGE, TARGET, RMAX, SCF, ATOMS, CONTROL, PRINT)
- Atomic coordinate format and ipot definitions
- Absorption edge types and selection

## Calculation Types

FEFF supports multiple calculation modes. See [references/02-calculation-types.md](references/02-calculation-types.md) for:

- **XANES**: Near-edge spectra with SCF convergence
- **EXAFS**: Extended spectra with NLEG path control
- **DOS**: Local density of states via LDOS keyword
- **Multiple Scattering**: NLEG configuration for complex paths

## Structure Input

Preparing atomic coordinates for FEFF. See [references/03-structure-input.md](references/03-structure-input.md) for:

- CIF to FEFF conversion via Artemis
- Cluster radius selection (RMAX)
- Coordinate unit conventions (Angstrom/Bohr)

## Advanced Features

Extending FEFF capabilities. See [references/04-advanced-features.md](references/04-advanced-features.md) for:

- Temperature effects: DEBYE keyword with Debye temperature
- Spin polarization: SPIN keyword
- Exchange potentials: Hedin-Lundqvist, Dirac-Hara, Overhauser
- Screening corrections: CORRECTIONS keyword
- Parallel computing: MPI configuration

## Error Recovery

Troubleshooting common FEFF issues. See [references/error-recovery.md](references/error-recovery.md) for:

| Issue | Solution |
|-------|----------|
| Cluster too small | Increase RMAX |
| SCF not converging | Adjust SCF/TOLERANCE parameters |
| Memory issues | Reduce RMAX or NLEG |
| Path errors | Verify ATOMS list, adjust CRITERIA |

## Templates

Template files in [assets/templates/](assets/templates/) provide starting points for common calculations:

| Template | Purpose |
|----------|---------|
| [`feff.inp`](assets/templates/feff.inp) | Fe K-edge XANES calculation with SCF |
| [`xanes.inp`](assets/templates/xanes.inp) | Combined EXAFS+XANES calculation |
| [`exafs.inp`](assets/templates/exafs.inp) | Cu K-edge EXAFS with DEBYE thermal effects |
| [`dos.inp`](assets/templates/dos.inp) | DOS calculation with LDOS |
| [`feff_slurm.sh`](assets/templates/feff_slurm.sh) | SLURM job submission script |

## Output Files

| File | Description |
|------|-------------|
| `xmu.dat` | Absorption coefficient spectrum |
| `chi.dat` | Normalized EXAFS signal |
| `paths.dat` | List of scattering paths |
| `feffNNNN.dat` | Path-specific amplitude and phase |
| `ldos.dat` | Local density of states |

## Guardrails

- Do not guess absorbing atom index; always verify TARGET corresponds to the correct atom in ATOMS list.
- Do not set RMAX too small for the phenomena of interest; EXAFS typically needs larger clusters than XANES.
- Do not run FEFF on login nodes; submit via the SLURM script template.
- Do not forget to check SCF convergence before interpreting results.

## Required Output

Always report:

- Calculation type and method (EXAFS/XANES/NEXAFS/DOS)
- Absorption edge and absorbing atom
- Cluster radius and number of atoms
- SCF convergence status
- Key output files (.dat, .chi, .mu)
