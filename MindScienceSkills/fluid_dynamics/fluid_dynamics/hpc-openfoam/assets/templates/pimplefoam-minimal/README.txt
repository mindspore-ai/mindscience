Minimal transient incompressible OpenFOAM-11 case scaffold using PIMPLE algorithm.

Contents:
- 0/U
- 0/p
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
   - pimpleFoam
   - foamRun -solver incompressibleFluid

Transient controls:
- adjustTimeStep is enabled; deltaT is a starting value.
- maxCo limits the Courant number to 1.0 by default.
- Raise writeInterval once the case is stable.

PIMPLE tuning notes:
- Increase nOuterCorrectors if the pressure-velocity coupling is slow to converge per time step.
- Increase nNonOrthogonalCorrectors only if mesh quality demands it.
- Do not inflate all corrector counts simultaneously before diagnosing the actual bottleneck.

If you replace the mesh, keep patch names consistent across:
- constant/polyMesh/boundary
- all files in 0/
