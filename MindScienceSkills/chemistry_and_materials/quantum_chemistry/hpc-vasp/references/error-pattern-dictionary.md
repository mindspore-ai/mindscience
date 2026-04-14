# VASP Error Pattern Dictionary

## Contents

- Input-set mismatch patterns
- SCF instability patterns
- Ionic relaxation patterns
- DOS and band handoff patterns

## Input-set mismatch patterns

### Pattern ID: `VASP_SPECIES_ORDER_MISMATCH`
- **Likely symptom**: run starts from a formally valid input set but yields physically nonsensical species assignment or immediately suspicious output interpretation
- **Root cause**: `POSCAR` species order and concatenated `POTCAR` order do not match
- **First checks**:
  - inspect species order in `POSCAR`
  - inspect concatenation order in `POTCAR`
  - inspect whether later stage files were copied from a different composition
- **Primary fix**: rebuild `POTCAR` in the exact `POSCAR` species order

### Pattern ID: `VASP_STAGE_TAG_MISMATCH`
- **Likely symptom**: a supposed static, DOS, or bands stage still behaves like an ionic optimization or vice versa
- **Root cause**: stage intent and `INCAR` tags do not match
- **First checks**:
  - verify `IBRION`
  - verify `NSW`
  - verify `ICHARG` or restart logic for follow-on stages
- **Primary fix**: replace the stage with a coherent stage-specific `INCAR` instead of patching single tags in place

## SCF instability patterns

### Pattern ID: `VASP_METAL_SMEARING_MISMATCH`
- **Likely symptom**: oscillatory SCF, slow convergence, or obviously poor occupation behavior
- **Root cause**: metallic versus insulating smearing logic is wrong
- **First checks**:
  - inspect `ISMEAR`
  - inspect `SIGMA`
  - inspect whether the system is metallic or gapped
- **Primary fix**: choose smearing consistent with the electronic class before tuning advanced convergence settings

### Pattern ID: `VASP_CUTOFF_OR_KMESH_TOO_WEAK`
- **Likely symptom**: unstable totals, inconsistent energies between stages, or poor SCF robustness
- **Root cause**: insufficient `ENCUT` or insufficient k-point density
- **First checks**:
  - inspect `ENCUT`
  - inspect KPOINTS density
  - inspect pseudo family expectations
- **Primary fix**: increase basis or sampling quality before changing many SCF algorithm tags

## Ionic relaxation patterns

### Pattern ID: `VASP_RELAXATION_TOO_AGGRESSIVE`
- **Likely symptom**: ionic steps bounce, forces remain erratic, or structural updates look unstable
- **Root cause**: relaxation settings are too aggressive for the starting geometry
- **First checks**:
  - inspect `IBRION`
  - inspect `POTIM`
  - inspect `ISIF`
  - inspect starting geometry reasonableness
- **Primary fix**: switch to a more conservative relaxation setup and verify the structure

### Pattern ID: `VASP_WRONG_RELAXATION_SCOPE`
- **Likely symptom**: cell changes when only ions should move, or geometry appears frozen when cell relaxation was intended
- **Root cause**: `ISIF` does not match the requested relaxation scope
- **First checks**:
  - inspect whether ions-only or ions-plus-cell behavior was intended
  - inspect current `ISIF`
- **Primary fix**: set `ISIF` from stage intent, then re-run the relaxation

## DOS and band handoff patterns

### Pattern ID: `VASP_DOS_WITH_WRONG_KPOINT_MODE`
- **Likely symptom**: poor DOS quality or stage intent inconsistent with the k-point file
- **Root cause**: DOS stage is using a band-path KPOINTS file or an overly weak mesh
- **First checks**:
  - inspect KPOINTS mode
  - inspect whether DOS or bands was the actual target
- **Primary fix**: use a dense uniform mesh for DOS

### Pattern ID: `VASP_BANDS_WITH_WRONG_HANDOFF`
- **Likely symptom**: bands stage fails or is inconsistent with prior SCF results
- **Root cause**: line-mode KPOINTS or restart logic not aligned with the prior converged state
- **First checks**:
  - inspect whether the source SCF was converged
  - inspect `ICHARG` or related handoff logic
  - inspect line-mode KPOINTS path
- **Primary fix**: rebuild the bands stage from a known-good static source state
