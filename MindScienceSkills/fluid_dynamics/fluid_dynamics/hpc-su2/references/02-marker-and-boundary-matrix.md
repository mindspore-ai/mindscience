# SU2 Marker And Boundary Matrix

## Contents

- Marker-name contract
- Common boundary categories
- Marker-selection matrix

## Marker-name contract

Mesh markers and config marker references must match exactly.

Treat marker names as a schema shared by:

- the mesh
- boundary-condition sections
- monitoring and output definitions

If a marker name differs in spelling or case, fix that first.

## Common boundary categories

Typical SU2 marker families include:

- inlet
- outlet
- farfield
- wall
- symmetry
- monitoring markers for forces and moments

Use the marker type that matches the physical role, not whichever keyword feels closest.

## Marker-selection matrix

| Physical role | Typical SU2 treatment |
| --- | --- |
| solid body | wall-style marker |
| far-field outer boundary | farfield-style marker |
| inlet with prescribed state | inlet marker with matching thermodynamic data |
| outlet | outlet or pressure-style exit marker |
| force monitoring surface | marker listed in monitoring output |

If the case asks for lift or drag, keep the monitored body marker explicit in the config.
