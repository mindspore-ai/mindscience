# DOLFINx Parallel And MPI Caveats

## Contents

- Ownership and marked entities
- Boundary marking in parallel
- PETSc options consistency
- Practical parallel checklist

## Ownership and marked entities

DOLFINx is MPI-first. Mesh ownership matters for:

- located entities
- located dofs
- IO
- refinement behavior

Do not assume a serial marking strategy is automatically parallel-safe.

## Boundary marking in parallel

The official `locate_entities_boundary` documentation explicitly warns:

- for vertices and edges, not all exterior-boundary entities are necessarily found on a given rank
- downstream dof-location logic typically needs parallel communication awareness

Inference for the skill:

- facet-level workflows are safer than vertex-level shortcuts
- if a boundary condition seems to disappear in parallel, inspect ownership and entity dimension before changing the PDE

## PETSc options consistency

From the official PETSc interface guidance:

- internally created PETSc objects have unique options prefixes
- options must be set consistently across MPI ranks

Do not hard-code guessed prefixes or let different ranks see different solver settings.

## Practical parallel checklist

Before trusting a parallel run:

1. confirm mesh and tags are distributed as intended
2. confirm boundary selection is parallel-safe
3. confirm PETSc options are consistent across ranks
4. confirm output paths and writer semantics make sense in MPI
5. if debugging, reduce to a robust solver configuration before scaling out
