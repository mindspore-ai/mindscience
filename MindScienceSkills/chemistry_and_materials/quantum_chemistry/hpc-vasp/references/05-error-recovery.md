# VASP Error Recovery

## Contents

- input-set mismatches
- SCF failures
- ionic relaxation failures
- workflow handoff failures

## Input-set mismatches

If VASP fails before or at startup:

1. verify `POSCAR` species order and count
2. verify `POTCAR` element order
3. verify `KPOINTS` mode fits the stage
4. verify `INCAR` tags are coherent for the stage

## SCF failures

If SCF oscillates or stalls:

1. inspect metallic versus insulating smearing logic
2. inspect `ENCUT` and k-point density
3. inspect initial magnetization assumptions if spin-polarized
4. then inspect electronic algorithm choices

## Ionic relaxation failures

If ionic optimization is unstable:

1. inspect starting structure quality
2. inspect `IBRION`, `NSW`, `ISIF`
3. use a more conservative relaxation setup before chasing subtle tags

## Workflow handoff failures

If DOS or bands look wrong:

1. verify the prior stage was converged
2. verify KPOINTS mode changed appropriately for the new stage
3. verify the right structure and charge-density context were carried forward
