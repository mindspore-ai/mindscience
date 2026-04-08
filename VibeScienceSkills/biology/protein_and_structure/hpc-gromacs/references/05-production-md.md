# Production MD

## When to Start Production

Production MD follows successful NPT equilibration. The system should have:
- Stable temperature at target
- Stable pressure at target
- Correct density
- No steric clashes or bad geometry

## Standard Production MDP

```mdp
; Production MD
integrator              = md-vv
dt                      = 0.002          ; 2 fs timestep
nsteps                  = 5000000       ; 10 ns

; Temperature coupling (should match NPT)
tcoupl                  = v-rescale
tc-grps                 = Protein Water_and_ions
tau-t                   = 0.1 0.1
ref-t                   = 300 300

; Pressure coupling
pcoupl                  = Parrinello-Rahman
pcoupltype              = isotropic
tau-p                   = 1.0
ref-p                   = 1.0
compressibility         = 4.5e-5
refcoord-scaling        = com

; Electrostatics
coulombtype             = PME
rcoulomb                = 1.0
rvdw                    = 1.0
vdwtype                 = Cut-off

; Constraints
constraints             = h-bonds
constraint-algorithm     = lincs

; Output
nstxout                 = 0             ; Full coords rarely needed
nstvout                 = 0
nstenergy               = 50000         ; 100 ps
nstlog                  = 50000
nstxout-compressed      = 10000         ; 20 ps, compressed
compressed-x-precision   = 1000
```

## Trajectory Output Strategy

| Format | Precision | Size | Use |
|--------|-----------|------|-----|
| `.trr` | Full | Very large | When needing exact coordinates every frame |
| `.xtc` | ~0.001 nm | Small | Standard production storage |
| `.pdb` | ~0.001 nm | Large | Single frames for analysis |

For most production runs, use compressed XTC with `nstxout-compressed`.

## Checkingpoints

Enable continuation in case of crashes:

```mdp
; Checkpointing
nstcalcenergy           = 100
nstenergy               = 10000
```

Resume from checkpoint:

```bash
gmx mdrun -cpi md.cpt -deffnm md
```

Append to existing trajectory:

```bash
gmx mdrun -cpi md.cpt -append yes -deffnm md
```

## PME (Particle Mesh Ewald) Settings

For electrostatics in periodic systems:

```mdp
coulombtype             = PME
rcoulomb                = 1.0           ; Usually same as rvdw
fourierspacing         = 0.12          ; 0.1-0.16 typical
pme-order              = 4             ; 4 is standard
ewald-rtol             = 1e-5
```

| Parameter | Effect | Tradeoff |
|-----------|--------|----------|
| `fourierspacing` smaller | More accurate, slower | Accuracy vs speed |
| `pme-order` higher | Better interpolation, more memory | 4 is usually optimal |
| `ewald-rtol` smaller | More accurate, slower | |

## LINCS Constraints

```mdp
constraints             = h-bonds
constraint-algorithm     = lincs
lincs-iter              = 1
lincs-order             = 4
```

With `h-bonds` constrained, dt can be 2 fs.

**Without constraints**, dt must be 0.5-1 fs, which is much slower.

## Performance Tuning

### For Multiple Nodes (MPI)

```mdp
nstlist                 = 20            ; Rebuild neighbor list every 20 steps
ns-type                 = grid
pbc                     = xyz
```

### GPU Acceleration

Modern GROMACS uses GPUs automatically when available:

```bash
gmx mdrun -deffnm md -nb gpu
```

Or let GROMACS auto-detect:

```bash
gmx mdrun -deffnm md
```

### Thread-MPI (Single Node)

```bash
gmx mdrun -deffnm md -ntomp 8 -npme 0
```

## Running Production

```bash
gmx grompp -f md.mdp -c npt.gro -t npt.cpt -p system.top -o md.tpr

# Single node
gmx mdrun -v -deffnm md

# Multi-node
srun -n 32 gmx mdrun -deffnm md
```

## Production Duration

| System Type | Minimum | Good | Excellent |
|-------------|---------|------|-----------|
| Small protein | 10 ns | 100 ns | 1 μs |
| Large protein | 10 ns | 100 ns | 1 μs |
| Membrane protein | 10 ns | 100 ns | 1 μs |
| Folding simulation | 1 μs | 10 μs | 100 μs+ |
| Drug binding | 100 ns | 1 μs | 10 μs+ |

## Common Production Problems

| Symptom | Cause | Fix |
|---------|-------|-----|
| System explodes | Bad NPT equilibration | Return to NPT |
| Energy drift | Not well equilibrated | Check NPT was stable |
| Pressure oscillations | tau-p too small | Increase tau-p |
| LINCS warnings | Bad geometry | Return to EM |
| "Planet" artifact | PBC issues in trajectory | Use `gmx trjconv -pbc mol` |
