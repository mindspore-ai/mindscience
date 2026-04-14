# OpenFOAM Mesh And `blockMeshDict` Manual

## Contents

- Mesh-generation workflow
- `blockMeshDict` anatomy
- Vertex ordering rules
- `checkMesh` workflow
- Surface and mesh debugging

## Mesh-generation workflow

Use this sequence for structured or semi-structured cases:

1. define geometry intent
2. write `system/blockMeshDict` or prepare another meshing route
3. run `blockMesh`
4. run `checkMesh`
5. only then trust numerics or boundary-condition debugging

If the mesh is invalid, stop. Do not tune `fvSchemes` to compensate for a broken grid.

## `blockMeshDict` anatomy

The core sections are:

- `convertToMeters`
- `vertices`
- `blocks`
- `edges`
- `boundary`
- `mergePatchPairs`

Each `hex` block maps:

- 8 vertex indices
- cell counts in three directions
- grading controls

Keep mesh topology simple while the physics and BCs are still under construction.

## Vertex ordering rules

The official `blockMesh` manual page is explicit that local vertex ordering matters.

Practical rule:

- the first 4 vertices define one face
- the next 4 define the opposite face
- the local numbering must match OpenFOAM's expected orientation

If vertex ordering is wrong, the case may build a mesh with inverted faces or negative-volume-like pathologies.

Treat face winding in the `boundary` section with the same care. Outward-facing normals are not optional.

## `checkMesh` workflow

The official `checkMesh` manual exposes useful operational modes:

- `-allGeometry`
- `-allTopology`
- `-meshQuality`
- `-writeAllFields`
- `-writeSets`
- `-parallel`

Use these deliberately:

- `-allTopology` when topology errors are suspected
- `-meshQuality` when a custom `meshQualityDict` is in play
- `-writeAllFields` or `-writeSets` when the user needs a visual debugging trail
- `-parallel` only when the decomposition is already trustworthy

Inference from the docs:

- `checkMesh` is not just a pass/fail gate
- it can also produce debugging artifacts that point to where the mesh is weak

## Surface and mesh debugging

Useful official tools:

- `surfaceCheck` for surface geometry and topology before meshing
- `checkMesh` after volume mesh creation

If the geometry source is suspicious:

1. validate the surface first
2. then create the volume mesh
3. then validate the volume mesh

Do not wait until solver divergence to discover a bad input surface.
