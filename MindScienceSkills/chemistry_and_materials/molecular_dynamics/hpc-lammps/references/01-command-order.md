# Command Order (The Law)

LAMMPS is extremely sensitive to command order. Violations cause silent
failures or garbage results.

## The Eight Stages

```
STAGE 1: Initialization        ← units, atom_style, boundary
STAGE 2: System Definition     ← read_data OR create_box + create_atoms
STAGE 3: Force Fields         ← pair_style, pair_coeff, bond_style, etc.
STAGE 4: Neighbor Settings     ← neighbor, neigh_modify
STAGE 5: Minimization          ← minimize (optional but recommended)
STAGE 6: Velocity Setup        ← velocity (optional)
STAGE 7: Fixes & Computes     ← fix, compute
STAGE 8: Run                   ← run, dump, thermo
```

## Stage 1: Initialization

```lammps
units metal           # or lj, real, cgs, etc.
atom_style atomic     # or charge, bond, full, etc.
boundary p p p        # p = periodic, f = fixed, s = shrink-wrap
```

**Rule:** These must appear before anything else that defines the system.

## Stage 2: System Definition

Choose ONE method:

### Method A: read_data

```lammps
read_data system.data
```

### Method B: create_box + create_atoms

```lammps
region mybox block 0 10 0 10 0 10
create_box 1 mybox

lattice fcc 3.615
region atoms block 0 10 0 10 0 10
create_atoms 1 region atoms
```

**Rule:** Do NOT mix `read_data` with `create_box`/`create_atoms`.

## Stage 3: Force Fields

```lammps
pair_style lj/cut 2.5
pair_coeff * * 1.0 1.0 2.5

# For EAM:
pair_style eam
pair_coeff * * Cu_u3.eam Cu
```

**Rule:** `pair_coeff` atom types must match those in the data file or
`create_atoms`.

## Stage 4: Neighbor Settings

```lammps
neighbor 0.3 bin
neigh_modify every 1 delay 0 check yes
```

## Stage 5: Minimization

```lammps
minimize 1.0e-4 1.0e-6 1000 10000
```

Minimization is strongly recommended before dynamics.

## Stage 6: Velocity (Optional)

```lammps
velocity all create 300.0 4928459 mom yes rot yes
```

## Stage 7: Fixes

```lammps
fix 1 all nvt temp 300.0 300.0 100.0
```

## Stage 8: Run

```lammps
thermo 100
dump 1 all atom 1000 dump.lammpstrj
run 10000
```

## Common Order Violations

| Violation | Symptom | Fix |
|-----------|---------|-----|
| `read_data` before `units` | Error or wrong interpretation | Move `units` before `read_data` |
| `pair_coeff` before `pair_style` | "Pair style not set" error | Move `pair_style` first |
| `create_atoms` before `lattice` | Wrong atom positions | Define `lattice` before `create_atoms` |
| `fix` before `pair_style` | Various | Put all initialization first |
| `run` before `pair_style` | "No fixes" or crash | Set up all fixes before `run` |

## Complete Example

```lammps
# Stage 1: Initialization
units metal
atom_style atomic
boundary p p p

# Stage 2: System
read_data Cu.data

# Stage 3: Force fields
pair_style eam
pair_coeff * * Cu_u3.eam

# Stage 4: Neighbor
neighbor 0.3 bin
neigh_modify every 1 delay 0 check yes

# Stage 5: Minimization
minimize 1.0e-4 1.0e-6 1000 10000

# Stage 6: Velocity
velocity all create 300.0 4928459

# Stage 7: Fixes
fix 1 all npt temp 300.0 300.0 100.0 iso 0.0 0.0 1000.0

# Stage 8: Run
thermo 100
dump 1 all atom 1000 dump.lammpstrj
run 10000
```
