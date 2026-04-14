Minimal transient incompressible multiphase OpenFOAM-11 case scaffold using interFoam (VOF).

Contents:
- 0/U, 0/p, 0/p_rgh, 0/alpha.water
- constant/g
- constant/transportProperties
- constant/phaseProperties
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
   - interFoam
   - foamRun -solver incompressibleVoF

Multiphase controls:
- maxCo and maxAlphaCo both limited to 1.0 for conservative startup.
- Adjust deltaT or the max limits once the interface is stable.
- interfaceCompression is enabled in fvSchemes div(phi,alpha) for stability.

Patch names must be consistent across:
- constant/polyMesh/boundary
- all files in 0/

Common failure modes:
- Interface smearing or blow-up: lower deltaT, tighten maxAlphaCo.
- Pressure oscillation at interface: verify p_rgh boundaries match hydrostatic expectations.
- Do NOT add alpha.water to a single-phase case without also switching solver family.
