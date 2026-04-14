# FEniCS And DOLFINx PETSc Solver Playbook

## Contents

- Linear problems
- Nonlinear problems
- PETSc option strategy
- Block and nested systems

## Linear problems

For DOLFINx, the high-level `LinearProblem` class is the standard choice for many linear variational problems.

Official guidance highlights:

- it wraps PETSc KSP
- it handles PETSc object lifetime automatically
- PETSc options are passed through `petsc_options`

Default KSP behavior is typically GMRES with ILU-like preconditioning. For a robust direct solve on moderate problems, the docs show LU with MUMPS as a standard pattern.

Use a direct solver when:

- the model is moderate in size
- debugging formulation correctness is more important than scalability
- you want to remove iterative-solver uncertainty during early development

Typical robust direct options shape:

```python
petsc_options = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}
```

Use that as a debugging baseline, then replace it only after the formulation is trustworthy.

## Nonlinear problems

For nonlinear variational problems in DOLFINx, use the high-level PETSc nonlinear problem support rather than hand-rolling Newton loops unless there is a clear reason.

Official guidance notes:

- the nonlinear interface is built on PETSc SNES
- PETSc object lifetime is handled automatically
- a robust documented option set uses LU via MUMPS with SNES backtracking line search

Use that robust pattern first when:

- the residual is sensitive to startup values
- the Jacobian is difficult
- you are still validating the formulation

Typical nonlinear option shape:

```python
petsc_options = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "bt",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}
```

This is not the only valid choice, but it is a high-confidence starting point from the official nonlinear examples.

If the problem is fundamentally nonlinear, do not try to rescue it by forcing a linear solve path.

## PETSc option strategy

Important operational rules from the DOLFINx PETSc interface:

- every internally created PETSc object gets a unique options prefix
- discover those prefixes programmatically rather than hard-coding guesses
- options must be consistent across ranks

Practical debugging sequence:

1. get the weak form correct
2. use robust direct options first
3. inspect convergence reason
4. only then move to scalable iterative or block-preconditioned designs

## Block and nested systems

DOLFINx explicitly supports nested and block-structured problems.

Use block or nested formulations when:

- the PDE is mixed
- different fields need different preconditioning logic
- you want a principled preconditioner rather than a monolithic black-box KSP

Official mixed demos show that blocked systems are not an advanced luxury. They are often the correct representation for mixed Poisson and related systems.
