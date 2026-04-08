Minimal steady incompressible OpenFOAM-11 case scaffold — laminar flow (no turbulence fields).

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
   - simpleFoam
   - foamRun -solver incompressibleFluid

If you replace the mesh, keep patch names consistent across:
- constant/polyMesh/boundary
- all files in 0/
