# Units and Timestep

## Unit System is Mandatory

The unit system defines:
- Length (Å, nm, m)
- Mass (g/mol, kg, amu)
- Time (fs, ps, s)
- Energy (eV, kcal/mol, J)
- Force (eV/Å, kcal/mol-Å, nN)

**Never copy a timestep value from one unit system to another.**

## Available Unit Systems

| Units | Length | Time | Energy | Force | Typical Use |
|-------|--------|------|--------|-------|-------------|
| `lj` | σ | σ/√ε/m | ε | ε/σ | Reduced LJ units |
| `metal` | Å | ps | eV | eV/Å | Metals, fs-resolution |
| `real` | Å | fs | kcal/mol | kcal/mol-Å | Organic molecules |
| `cgs` | cm | s | erg | dyne | Rarely used |
| `si` | m | s | J | N | Rarely used |

## Metal vs Real Units

### Metal Units (For Metallic Systems)

```lammps
units metal

# Timestep: 0.001-0.005 ps = 1-5 fs
# Typical: 0.002 ps (2 fs)

# Example with 2 fs timestep:
timestep 0.002
```

### Real Units (For Organic/ Biomolecular)

```lammps
units real

# Timestep: 0.5-1.0 fs for constrained bonds
# 1.0 fs with h-bonds constrained to H

timestep 1.0
```

## Timestep Selection Rules

| System | Constraints | Max Timestep | Notes |
|--------|-------------|--------------|-------|
| Pure LJ fluid | None | 0.002 (reduced) | Depends on tau |
| Metal (EAM) | None | 0.002 ps (2 fs) | No constraints in metals |
| Organic | H-bonds constrained | 1.0-2.0 fs | Use `fix shake` or `fix rattle` |
| Water | Constrained | 2.0 fs | Standard TIP3P |
| Coarse-grained | None | 5-20 fs | Depends on potential |

**Rule:** Start conservative (smaller dt) and increase if the system is stable.

## Reduced Units (lj)

In reduced LJ units, everything is dimensionless:

```lammps
units lj
variable sigma equal 1.0
variable epsilon equal 1.0
variable m equal 1.0

timestep 0.005    # ~0.005 * tau
```

Typical reduced timestep: 0.005-0.01 τ.

## Time in Different Units

| Quantity | metal | real |
|---------|-------|------|
| 1 second | 1e12 fs | 1e15 fs |
| 1 picosecond | 1 fs | 1000 fs |
| 1 nanosecond | 1000 fs | 1e6 fs |

## Temperature and Pressure Units

| Unit | Temperature | Pressure |
|------|-----------|----------|
| `metal` | K | atm (converts to Bar) or bar |
| `real` | K | atm |
| `lj` | ε/kB | ε/σ³ |

## Verifying Unit Consistency

In your script, document units explicitly:

```lammps
# Cu melting simulation
# Units: metal
# Timestep: 0.002 ps (2 fs)
# Temperature: 300-1350 K
# Pressure: 1 atm

units metal
timestep 0.002
```

## Damping Parameters (Time Constants)

| Parameter | metal | real |
|-----------|-------|------|
| `Tdamp` (thermostat) | 100 * dt = 0.2 ps | 100 * dt = 100 fs |
| `Pdamp` (barostat) | 1000 * dt = 2 ps | 1000 * dt = 1 ps |

Tdamp and Pdamp in reduced time units must be large enough to couple
correctly. 100×dt is a standard starting point.

## Timestep Errors

| Symptom | Cause | Fix |
|---------|-------|-----|
| "Lost atoms" immediately | Timestep too large | Reduce timestep |
| Energy drift | Timestep too large | Reduce timestep |
| Temperature oscillations | Tdamp too small | Increase Tdamp |
| Pressure oscillations | Pdamp too small | Increase Pdamp |
