# Equilibration (NVT and NPT)

## Why Two Equilibration Stages?

**NVT (Canonical):** System at fixed N, V, T
- Purpose: Bring temperature to target value
- Stabilize kinetic energy distribution
- No pressure control yet

**NPT (Isothermal-Isobaric):** System at fixed N, P, T
- Purpose: Bring pressure (and density) to target value
- Allow box size to equilibrate
- Essential for getting correct density

**Never run production MD without NPT.** The box size from NVT is not
equilibrated for pressure.

## NVT Equilibration

### When to Use

After successful EM, before NPT.

### Standard NVT MDP

```mdp
; NVT equilibration
integrator              = md-vv
dt                      = 0.002
nsteps                  = 50000         ; 100 ps

; Temperature coupling
tcoupl                  = V-rescale
tc-grps                 = Protein Water_and_ions
tau-t                   = 0.1 0.1
ref-t                   = 300 300

; Pressure coupling (off for NVT)
pcoupl                  = no

; Output
nstxout                 = 0
nstvout                 = 0
nstenergy               = 5000
nstlog                  = 5000
```

### Temperature Coupling Groups

| Group | Contents | Why Separate? |
|-------|----------|--------------|
| Protein | Protein atoms | Heat capacity differs from water |
| Water_and_ions | Solvent + ions | Bulk properties |
| Other | Ligands, membranes | If present |

**Rule:** `tc-grps` should match the physical components.

### Thermostat Choice

| Thermostat | Notes |
|------------|-------|
| `v-rescale` | Recommended. Stochastically modified Nosé-Hoover. |
| `nose-hoover` | Deterministic. Good for NPT. |
| `berendsen` | Old, not recommended for production. OK for quick equilibration. |

## NPT Equilibration

### When to Use

After NVT, before production MD.

### Standard NPT MDP

```mdp
; NPT equilibration
integrator              = md-vv
dt                      = 0.002
nsteps                  = 50000         ; 100 ps

; Temperature coupling
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

; Output
nstxout                 = 0
nstvout                 = 0
nstenergy               = 5000
```

### Pressure Coupling Types

| Type | Use Case |
|------|----------|
| `isotropic` | Liquids, small molecules, globular proteins |
| `semiisotropic` | Membranes (x,y different from z) |
| `anisotropic` | Systems with very different box dimensions |

### Barostat Choice

| Barostat | Notes |
|----------|-------|
| `Parrinello-Rahman` | Good for production. Allows box oscillations. |
| `Berendsen` | Old, good for quick equilibration. |
| `c-rescale` | For NPT with v-rescale thermostat. |

## NPT Box Size Issues

If the box shrinks/grows excessively:

| Problem | Fix |
|---------|-----|
| Box shrinks a lot | tau-p too small, increase to 2-5 |
| Box expands a lot | System not neutral, add ions |
| Box oscillates | tau-p too small, increase |

## Running Equilibration

```bash
# NVT
gmx grompp -f nvt.mdp -c em.gro -r em.gro -p system.top -o nvt.tpr
gmx mdrun -v -deffnm nvt

# NPT
gmx grompp -f npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p system.top -o npt.tpr
gmx mdrun -v -deffnm npt
```

`-r em.gro` (reference) and `-t nvt.cpt` (checkpoint) ensure continuation.

## Checking Equilibration

### NVT: Check Temperature

```bash
gmx energy -f nvt.edr -o temperature.xvg
```

- Temperature should oscillate around `ref-t`
- Average should be close to `ref-t`
- Discard first ~10-20 ps as equilibration

### NPT: Check Pressure and Density

```bash
gmx energy -f npt.edr -o pressure.xvg
gmx energy -f npt.edr -o density.xvg
```

- Pressure should oscillate around `ref-p`
- Density should stabilize (water ~1000 kg/m³ at 300K)

## Common Equilibration Problems

| Symptom | Cause | Fix |
|---------|-------|-----|
| Temperature too high/low | tau-t wrong | Adjust tau-t (0.1 is standard) |
| Pressure not stabilizing | tau-p too aggressive | Increase tau-p to 2-5 |
| Density wrong | Force field/water mismatch | Verify water model |
| System exploding | Bad structure from EM | Return to EM, check structure |
| "Shake cannot converge" | Bad geometry | EM may not have converged |
