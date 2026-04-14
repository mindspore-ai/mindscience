Minimal steady incompressible OpenFOAM-11 case scaffold — RAS kEpsilon turbulence.

Contents:
- 0/U, 0/p, 0/k, 0/epsilon, 0/nut
- constant/transportProperties
- constant/turbulenceProperties
- system/blockMeshDict
- system/controlDict
- system/decomposeParDict
- system/fvSchemes
- system/fvSolution

This template is self-contained and can run as-is:
1. blockMesh
2. checkMesh
3. Run one of:
   - simpleFoam
   - foamRun -solver incompressibleFluid

Patch names must be consistent across:
- constant/polyMesh/boundary
- all files in 0/

Startup strategy:
- Conservative (Gauss linearUpwind) used by default for div(phi,U).
- After residuals plateau, upgrade div(phi,U) to Gauss linear for second-order accuracy.
- If bounding epsilon appears early, tighten the epsilon inlet value or add bounding epsilon to fvSchemes divSchemes.
