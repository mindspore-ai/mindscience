# Thermostats and Barostats

## Integration Fixes

LAMMPS has fixes that both integrate and control ensembles:

| Fix | Integrates | Thermostat | Barostat | Notes |
|-----|-----------|------------|----------|-------|
| `nve` | Yes | No | No | Pure NVE |
| `nvt` | Yes | Yes | No | NVT |
| `npt` | Yes | Yes | Yes | NpT |
| `nph` | Yes | No | Yes | NpH |

## NVE (Microcanonical)

Energy-conserving, no thermostat or barostat:

```lammps
fix 1 all nve
```

Use NVE after NVT/NPT to observe isolated system behavior.

## NVT (Canonical)

Fixed N, V, T:

```lammps
fix 1 all nvt temp 300.0 300.0 100.0
```

Format: `temp T_initial T_target T_damp`

Tdamp (damping parameter):
- Too small: Temperature oscillations
- Too large: Slow to reach target
- Standard: 100 × dt

| Units | Tdamp = 100 × dt |
|-------|------------------|
| metal | 0.2 ps |
| real | 100 fs |

## NPT (Isothermal-Isobaric)

Fixed N, P, T:

```lammps
fix 1 all npt temp 300.0 300.0 100.0 iso 1.0 1.0 1000.0
```

Format: `temp T_initial T_target T_damp iso P_initial P_target P_damp`

For anisotropic pressure:

```lammps
fix 1 all npt temp 300.0 300.0 100.0 x 1.0 1.0 1000.0 z 1.0 1.0 1000.0
```

## NPH (NpH)

Fixed N, P, H (enthalpy):

```lammps
fix 1 all nph iso 1.0 1.0 1000.0
```

No thermostat. Barostat only.

## Langevin Thermostat

Stochastic thermostat, good for equilibrium sampling:

```lammps
fix 1 all langevin 300.0 300.0 100.0 48279
```

Format: `T_start T_stop T_damp seed`

Must be combined with NVE for integration:

```lammps
fix 1 all nve
fix 2 all langevin 300.0 300.0 100.0 48279
```

**Rule:** Never use two full integrators on the same atoms.

## Berendsen Thermostat

Simple, fast thermostat (not recommended for production):

```lammps
fix 1 all berendsen temp 300.0 300.0 100.0
```

Only use for quick equilibration. Produces incorrect ensemble.

## Nose-Hoover (nvt/npt)

The default for `nvt` and `npt`:

```lammps
fix 1 all nvt temp 300.0 300.0 100.0
```

Uses Nose-Hoover chains internally.

## Multiple Thermostats (for different groups)

```lammps
# Thermostat solute and solvent separately
fix 1 solute nvt temp 300.0 300.0 100.0
fix 2 solvent nvt temp 300.0 300.0 100.0
```

Each fix thermostatting different atom groups.

## Barostat Damping

Pdamp must be large enough to avoid pressure oscillations:

| Pdamp | metal | real |
|-------|-------|------|
| Standard | 1000 × dt | 1000 × dt |
| For oscillations | 5000 × dt | 5000 × dt |

Pdamp = 1000 × dt ≈ 1 ps is a good starting point.

## Pressure Coupling Types

| Type | Meaning | Use |
|------|---------|-----|
| `iso` | Isotropic (all directions same) | Liquids, cubic crystals |
| `aniso` | Anisotropic (each direction independent) | Non-cubic crystals |
| `x` | Only x-direction | Special cases |
| `tri` | Triclinic box | Arbitrary shapes |

## Temperature Groups

```lammps
group solute type 1 2 3
group solvent type 4 5

fix 1 solute nvt temp 300.0 300.0 100.0
fix 2 solvent nvt temp 300.0 300.0 100.0
```

Thermostatting groups separately maintains correct kinetic energy distribution.

## Common Ensemble Errors

| Symptom | Cause | Fix |
|---------|-------|-----|
| Temperature oscillating | Tdamp too small | Increase Tdamp |
| Temperature never reaches target | Tdamp too large | Decrease Tdamp |
| Pressure oscillating | Pdamp too small | Increase Pdamp |
| Box shrinking/growing | NPT with bad Pdamp | Adjust Pdamp |
| "Thermostat not supported" | Using wrong fix | Use correct NVT/NPT |
