# MDP Parameter Reference

## Parameter Categories

```
MDP Parameters
├── Integration      (dt, nsteps, integrator)
├── Temperature      (tcoupl, tau-t, ref-t)
├── Pressure        (pcoupl, tau-p, ref-p)
├── Electrostatics  (coulombtype, rcoulomb, PME settings)
├── Van der Waals  (vdwtype, rvdw, vdw-modifier)
├── Constraints     (constraints, lincs settings)
└── Output          (nstxout, nstenergy, etc.)
```

## Integration

| Parameter | Values | Meaning |
|-----------|--------|---------|
| `integrator` | `md`, `md-vv`, `steep`, `cg`, `sd` | Integration algorithm |
| `dt` | 0.0001-0.01 (ps) | Timestep. 0.002 (2 fs) is standard with constraints |
| `nsteps` | integer | Number of steps. Total time = dt × nsteps |

**Integrator options:**
- `md` — Standard molecular dynamics (old velocity Verlet)
- `md-vv` — Velocity Verlet (modern, energy conservation better)
- `steep` — Steepest descent minimization
- `cg` — Conjugate gradient minimization
- `sd` — Stochastic dynamics (Langevin)

## Temperature Coupling

| Parameter | Values | Meaning |
|-----------|--------|---------|
| `tcoupl` | `no`, `v-rescale`, `nose-hoover`, `berendsen` | Thermostat |
| `tc-grps` | group names | Groups to couple separately |
| `tau-t` | > 0 (ps) | Coupling time constant. 0.1 is standard |
| `ref-t` | temperature (K) | Target temperature |

**Groups should be physically meaningful:** Protein, Water_and_ions, etc.

## Pressure Coupling

| Parameter | Values | Meaning |
|-----------|--------|---------|
| `pcoupl` | `no`, `Parrinello-Rahman`, `Berendsen`, `c-rescale` | Barostat |
| `pcoupltype` | `isotropic`, `semiisotropic`, `anisotropic` | Box scaling |
| `tau-p` | > 0 (ps) | Coupling time constant |
| `ref-p` | pressure (bar) | Target pressure |
| `compressibility` | 4.5e-5 (1/bar) | Water compressibility |

**Box scaling:**
- `isotropic` — Same in all directions (liquids)
- `semiisotropic` — x,y different from z (membranes)
- `anisotropic` — All different (for anisotropic systems)

## Electrostatics

| Parameter | Values | Meaning |
|-----------|--------|---------|
| `coulombtype` | `PME`, `Cut-off`, `Ewald`, `Reaction-field` | Electrostatics method |
| `rcoulomb` | distance (nm) | Coulomb cutoff. 1.0 is standard |
| `fourierspacing` | 0.08-0.16 (nm) | PME grid spacing |
| `pme-order` | 3-6 | PME interpolation order. 4 is standard |
| `ewald-rtol` | 1e-6 to 1e-4 | Ewald/PME tolerance |

**Recommendation:** Always use PME for production. Cut-off only for testing.

## Van der Waals

| Parameter | Values | Meaning |
|-----------|--------|---------|
| `vdwtype` | `Cut-off`, `PME`, `Shift` | VdW method |
| `rvdw` | distance (nm) | VdW cutoff. Usually 1.0-1.2 |
| `vdw-modifier` | `None`, `Force-switch`, `LJ-PM E-shift` | Cut-off modifier |

**Common combination:** `rvdw = 1.0` with `vdwtype = Cut-off`.

## Constraints

| Parameter | Values | Meaning |
|-----------|--------|---------|
| `constraints` | `none`, `h-bonds`, `all-bonds`, `h-angles` | Constrained bonds |
| `constraint-algorithm` | `lincs`, `shake` | Constraint algorithm |
| `lincs-iter` | 1-4 | LINCS iterations |
| `lincs-order` | 4-8 | LINCS matrix order |

**Standard production:** `constraints = h-bonds`, `constraint-algorithm = lincs`, `lincs-iter = 1`.

## Output Control

| Parameter | Values | Meaning |
|-----------|--------|---------|
| `nstxout` | steps | Full coordinates output frequency |
| `nstvout` | steps | Full velocities output frequency |
| `nstxout-compressed` | steps | Compressed trajectory frequency |
| `nstenergy` | steps | Energy output frequency |
| `nstlog` | steps | Log file output frequency |
| `compressed-x-precision` | integer | XTC compression precision |

**Example:** `nstxout-compressed = 10000` + `dt = 0.002` = output every 20 ps.

## Neighbor Searching

| Parameter | Values | Meaning |
|-----------|--------|---------|
| `nstlist` | steps | Neighbor list update frequency |
| `ns-type` | `grid`, `simple` | Neighbor search algorithm |
| `pbc` | `xyz`, `xy`, `no` | Periodic boundary conditions |

**Standard:** `nstlist = 20`, `ns-type = grid`, `pbc = xyz`.

## Pull (Steered MD, Umbrella Sampling)

| Parameter | Values | Meaning |
|-----------|--------|---------|
| `pull` | `no`, `umbrella`, `constant-force`, `cm` | Pulling protocol |
| `pull-ncoords` | integer | Number of CVs |
| `pull-group1` | index group | First group |
| `pull-group2` | index group | Second group |
| `pull-k1` | force constant | Spring constant |
| `pull-rate1` | distance/time | Pulling rate |

For advanced sampling methods (umbrella sampling, metadynamics), use PLUMED
plugin instead of native GROMACS pull.

## Common .mdp Issues

| Issue | Symptom | Fix |
|-------|---------|-----|
| Timestep too large | LINCS warnings, explosion | Reduce dt to 0.001 or 0.0005 |
| tau-t too small | Temperature oscillations | Increase tau-t to 0.2-0.5 |
| tau-p too small | Pressure oscillations | Increase tau-p to 2-5 |
| Wrong units | Everything explodes | Verify dt in ps, not fs |
| Missing constraints | Very slow or dt must be small | Add `constraints = h-bonds` |
