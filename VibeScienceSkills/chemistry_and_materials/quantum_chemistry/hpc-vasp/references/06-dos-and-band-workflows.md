# VASP DOS And Band Workflows

## Contents

- stage ordering
- DOS workflow
- band-structure workflow
- k-point mode differences

## Stage ordering

Use a reliable stage chain:

1. relaxation if structure is not fixed
2. accurate static SCF
3. DOS or bands follow-on stage

Do not use a noisy relaxation stage as the source of final electronic analysis.

## DOS workflow

Typical DOS logic:

- start from a converged structure
- perform a high-quality static calculation
- use dense uniform k-point sampling suitable for DOS

Uniform sampling matters more than a symmetry-path k-point list in this stage.

## Band-structure workflow

Typical bands logic:

- start from a converged structure and electronic state
- prepare a band-path KPOINTS file
- run a dedicated follow-on stage

Do not confuse line-mode band paths with uniform SCF meshes.

## K-point mode differences

Use:

- Gamma or Monkhorst-Pack style meshes for SCF or static stages
- line-mode paths for band structures

If the user asks for both DOS and bands, they are usually separate downstream stages, not one shared KPOINTS file.
