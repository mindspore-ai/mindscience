# VASP Pseudopotential, K-Points, And Convergence

## Contents

- POTCAR coherence
- metallic versus insulating logic
- convergence troubleshooting matrix

## POTCAR coherence

The official VASP wiki treats `POTCAR` as a dataset choice tied to element order and setup assumptions.

Practical rules:

- keep element order consistent with `POSCAR`
- do not mix inconsistent potential families casually
- review recommended cutoff expectations when selecting the dataset

## Metallic versus insulating logic

High-value distinction:

- metals usually need smearing logic suited for partial occupancies
- insulators usually use a different occupation strategy

If the electronic structure class is wrong, `ISMEAR` and `SIGMA` choices often become the real problem before any advanced mixing logic.

## Convergence troubleshooting matrix

| Symptom | First things to inspect |
| --- | --- |
| SCF instability | `ISMEAR`, `SIGMA`, `ALGO`, `NELM`, initial magnetization assumptions |
| poor total-energy consistency | `ENCUT`, KPOINTS density, `PREC` |
| ionic relaxation behaving badly | `IBRION`, `POTIM`, `ISIF`, starting structure |

Do not tune every INCAR knob at once. Start with electronic class, cutoff, and k-points.
