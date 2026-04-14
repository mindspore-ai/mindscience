# VASP INCAR Tag Matrix

## Contents

- Stage-to-INCAR matrix
- Electronic-class-to-smearing matrix
- Relaxation-control matrix

## Stage-to-INCAR matrix

Use this as a fast lookup.

| Stage | High-value tags |
| --- | --- |
| geometry relaxation | `IBRION`, `NSW`, `ISIF`, `EDIFFG`, `POTIM` |
| static SCF | `EDIFF`, `ENCUT`, `ISMEAR`, `SIGMA`, `IBRION=-1`, `NSW=0` |
| DOS follow-on | static-like tags plus DOS-oriented k-point density |
| bands follow-on | static-like tags plus line-mode k-points and handoff settings |

If the stage and INCAR intent do not match this table, fix that mismatch before tuning convergence details.

## Electronic-class-to-smearing matrix

Based on the official VASP wiki guidance:

| System class | Typical smearing logic |
| --- | --- |
| insulator or semiconductor | `ISMEAR=0` with small `SIGMA`, or tetrahedron-style methods in appropriate static workflows |
| metal | metallic smearing such as Fermi or Methfessel-Paxton family, with care |
| high-quality static DOS workflow | tetrahedron-style methods on suitable meshes |

Important official warning:

- the default `ISMEAR=1` is not appropriate for insulators and semiconductors
- tetrahedron methods require a Gamma-centered k-mesh

## Relaxation-control matrix

| Goal | High-value tags |
| --- | --- |
| relax ions only | `IBRION`, `NSW`, `EDIFFG`, `ISIF` set to ion-only style |
| relax ions and cell | `ISIF` expanded to include cell degrees of freedom |
| static no-relax stage | `IBRION=-1`, `NSW=0` |

If a supposed static stage still has ionic motion enabled, correct that first.
