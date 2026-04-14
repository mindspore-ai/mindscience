# Neighbor Settings

## Why Neighbor Settings Matter

Neighbor lists determine which atom pairs are computed for pairwise
interactions. They are critical for:
- Numerical stability
- Performance
- Accuracy

**Bad neighbor settings cause lost atoms, wrong energies, and crashes.**

## Basic Neighbor Commands

```lammps
neighbor 0.3 bin
neigh_modify every 1 delay 0 check yes
```

| Command | Meaning |
|---------|---------|
| `neighbor 0.3 bin` | Rebuild if atom moves >0.3 Å |
| `neigh_modify every 1` | Check every step |
| `delay 0` | No delay before first rebuild |
| `check yes` | Only rebuild if atom moved > skin |

## Skin Distance

The skin distance determines how often neighbor lists are rebuilt:

| Skin Value | Rebuild Frequency | Memory | Safety |
|-----------|-----------------|--------|--------|
| Small (0.1-0.2) | More frequent | Less | Less safe |
| Standard (0.3) | Moderate | Moderate | Safe |
| Large (0.5+) | Less frequent | More | Safer but slower |

```lammps
neighbor 0.3 bin   # Standard for most systems
neighbor 0.5 bin   # For high-speed or hot systems
```

## Stability Scenarios

### Hot Systems

```lammps
neighbor 0.4 bin
neigh_modify every 1 delay 0 check yes
```

Larger skin for systems with high velocities (shock, impact).

### Dense/High-Pressure

```lammps
neighbor 0.3 bin
neigh_modify every 1 delay 0 check yes
```

Standard settings are usually fine.

### Deforming/Shearing

```lammps
neighbor 0.3 bin
neigh_modify every 1 delay 0 check yes
```

If deformation is extreme, increase skin or rebuild every step.

## Rebuild Frequency

### Every Step (Safest)

```lammps
neigh_modify every 1 delay 0 check yes
```

Recommended for:
- Shock simulations
- High-temperature systems
- Systems with fast-moving atoms

### Every Few Steps (Faster)

```lammps
neigh_modify every 5 delay 10 check yes
```

Only for stable, well-equilibrated systems.

## Pair Distance Cutoffs

The neighbor skin plus the pair cutoff must be larger than the interaction cutoff:

```lammps
pair_style lj/cut 2.5
neighbor 0.3 bin
```

Here: interaction cutoff = 2.5 Å, neighbor cutoff = 2.5 + 0.3 = 2.8 Å.

**Rule:** pair cutoff + skin must be < neighbor cutoff for safety.

## Multi-Level Neighbors

For large systems, use `neigh_modify` to exclude certain interactions:

```lammps
# Exclude 1-2 and 1-3 interactions (handled by bonds/angles)
neigh_modify exclude group int
```

This speeds up computation for molecular systems.

## Verifying Neighbor Settings

Monitor in the log file:

```
Neighbor list info...
  update every 1 steps, delay 0 steps, check yes
  max neighbors/atom: 2000
  cutoff 2.5
```

If `max neighbors/atom` approaches the limit, increase the limit or
adjust skin distance.

## Common Neighbor Errors

| Symptom | Cause | Fix |
|---------|-------|-----|
| "Lost atoms" | Atoms moved too far between rebuilds | Increase skin, rebuild every step |
| "Neighbor list overflow" | Too many neighbors per atom | Increase skin or use larger cutoff |
| "Too many neighbor bins" | System too heterogeneous | Adjust bin size or use `nsq` style |
| Energy drift | Skin too small causing missing interactions | Increase skin |
| Slow performance | Rebuilding too often | Tune skin and delay |

## Restarting with Different Settings

If restarting from a dump file:

```lammps
read_restart restart.file
neighbor 0.3 bin    # Can modify after restart
neigh_modify every 1 delay 0 check yes
```

Neighbor settings can be changed after `read_restart`.
