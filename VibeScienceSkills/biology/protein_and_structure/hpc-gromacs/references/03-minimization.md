# Energy Minimization

## Purpose

EM removes steric clashes and high-energy interactions before dynamics:

- Van der Waals overlaps
- Improper dihedral violations
- Bad bond lengths from structure building

**Do not skip EM.** Starting dynamics without EM risks exploding the system.

## EM Workflow

```bash
# 1. Create .tpr from system + em.mdp
gmx grompp -f em.mdp -c solvated_ions.gro -p system.top -o em.tpr

# 2. Run minimization
gmx mdrun -v -deffnm em

# 3. Check output
gmx energy -f em.edr -o potential.xvg
```

## Standard EM MDP

```mdp
; Energy minimization
integrator               = steep
nsteps                   = 50000
emtol                    = 1000.0
emstep                   = 0.01

; Output
nstlog                   = 1000
nstenergy               = 1000
```

| Parameter | Meaning | Typical Value |
|-----------|---------|---------------|
| `integrator` | Algorithm | `steep` (fast) or `cg` (slower, better) |
| `nsteps` | Max steps | 10,000-100,000 |
| `emtol` | Force convergence (kJ/mol/nm) | 100-1000 |
| `emstep` | Initial step size (nm) | 0.01 |

## Minimizer Choice

| Minimizer | Speed | Use When |
|-----------|-------|----------|
| `steep` | Fast | Most cases, first attempt |
| `cg` | Slower, more thorough | Steep fails to converge |
| `lbfgs` | Good for large systems | Conjugate gradient alternative |

**Typical approach:** First try `steep`, then `cg` if it doesn't converge.

## Checking EM Success

### Check potential energy

```bash
gmx energy -f em.edr -o potential.xvg
```

Look at potential energy:
- Should be **negative** (large negative for solvated systems)
- Should **decline** over iterations
- Final value should be stable (no sudden jumps)

### Check maximum force

```bash
gmx energy -f em.edr -o maxforce.xvg
```

- `F_max` should be below `emtol` at the end
- If F_max ≈ emtol, system is at acceptable energy

### Visual inspection

```bash
# Convert to trajectory and view
gmx trjconv -s em.tpr -f em.gro -o em.pdb -pbc mol
```

Check for:
- Water molecules inside the protein
- Clashing atoms
- Broken molecules

## EM Failures

| Symptom | Cause | Fix |
|---------|-------|-----|
| Energy increases | Step too large | Reduce `emstep` to 0.001-0.005 |
| Energy oscillates | Bad starting structure | Fix structure or use `cg` minimizer |
| F_max stays high | System is highly strained | May need longer minimizer, or structure issues |
| Segfault | Bad topology, corrupted file | Rebuild system or check .top |

## Using CG Minimizer

```mdp
integrator = cg
nsteps = 50000
emtol = 500.0
cg-steep = no
nstcgsteep = 1000
```

`cg-steep = no` = use pure CG without occasional steep steps.

## Re-running EM

If EM didn't converge:

```mdp
; Start from previous EM output
integrator = cg
nsteps = 50000
emtol = 500.0
init-lambda-state = 0
```

Use the last frame of the failed EM as input:

```bash
gmx grompp -f em2.mdp -c em.gro -t em.cpt -p system.top -o em2.tpr
```

## EM in Context

```
System built (solvated_ions.gro)
        ↓
    EM (em.tpr → em.gro)
        ↓
    NVT (nvt.tpr → nvt.gro)
        ↓
    NPT (npt.tpr → npt.gro)
        ↓
  Production (md.tpr → md.trr)
```

EM output (`em.gro`) is the input to NVT.
