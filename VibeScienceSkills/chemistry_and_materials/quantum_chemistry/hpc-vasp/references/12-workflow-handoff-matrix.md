# VASP Workflow Handoff Matrix

## Contents

- Stage-artifact matrix
- Restart-file matrix
- KPOINTS handoff matrix

## Stage-artifact matrix

| Completed stage | Primary artifact for next stage |
| --- | --- |
| relaxation | relaxed structure from `CONTCAR` |
| static SCF | trustworthy electronic state and output summaries |
| DOS preparation | dense static-like state suitable for DOS post-processing |
| bands preparation | converged prior state plus line-mode KPOINTS follow-on |

Use the right artifact for the right downstream stage. Do not feed a noisy intermediate structure into final analysis.

## Restart-file matrix

| File | Typical role |
| --- | --- |
| `CONTCAR` | structural handoff |
| `WAVECAR` | wavefunction restart candidate |
| `CHGCAR` | charge-density handoff candidate |
| `OUTCAR` | interpretation and audit trail |

Use `ISTART` and `ICHARG` consistently with the chosen handoff files.

## KPOINTS handoff matrix

| Workflow stage | Typical KPOINTS mode |
| --- | --- |
| routine SCF or relaxation | uniform mesh |
| high-quality static or DOS | dense uniform mesh |
| band structure | line-mode path |

If DOS and bands are both needed, they usually require different KPOINTS files.
