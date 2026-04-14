# Job Types

## Single Point (SP)

Energy calculation on a fixed geometry:

```gjf
#p B3LYP/6-311G(d,p)

Water single point energy

0 1
O   0.0  0.0  0.0
H   0.96  0.0  0.0
H  -0.24  0.93  0.0
```

Use when: You only need the energy, no geometry changes.

## Geometry Optimization (Opt)

Finds the minimum energy structure:

```gjf
#p B3LYP/6-31G(d,p) Opt
```

Gaussian optimizes by iteratively adjusting geometry until forces are small.

### Opt Options

| Option | Meaning |
|--------|---------|
| `Opt` | Default optimization |
| `Opt=ModRedundant` | Partial optimization (freeze some coordinates) |
| `Opt=CalcFC` | Calculate force constants at start |
| `Opt=CalcAll` | Calculate FC at every point |
| `Opt=TS` | Find transition state |
| `Opt=ZDO` | No geometry changes, just Z-matrix update |

### Transition State (TS) Optimization

```gjf
#p B3LYP/6-31G(d,p) Opt=TS
```

TS search requires a good starting guess (approximate saddle point).

## Frequency Calculation (Freq)

Vibrational analysis and thermochemical corrections:

```gjf
#p B3LYP/6-31G(d,p) Freq
```

Outputs:
- Zero-point energy (ZPE)
- Enthalpy corrections
- Entropy
- Heat capacity
- Thermodynamic properties

**Always run Freq at the same level as Opt** for accurate thermochemistry.

## Combined Opt + Freq

```gjf
#p B3LYP/6-31G(d,p) Opt Freq
```

Optimize, then compute frequencies at the optimized geometry.

**Important:** For a stable molecule, all frequencies should be real (>0).
Imaginary frequencies (negative in output) indicate a transition state.

## Potential Energy Surface (Scan)

Scan along one or more internal coordinates:

```gjf
#p B3LYP/6-31G(d,p) Scan

Molecule scan

0 1
C
C 1 1.54
```

For a relaxed scan along a bond:

```gjf
#p B3LYP/6-31G(d,p) Scan=ModRedundant

Bond scan

0 1
C
C 1 1.5
H 2 1.09 1 109.5
H 2 1.09 1 109.5 3 120.0
H 2 1.09 1 109.5 3 -120.0

1 2 0.05 10  # ModRedundant: vary distance atom1-atom2
```

## IRC (Intrinsic Reaction Coordinate)

Follow the reaction path from TS to reactants/products:

```gjf
#p B3LYP/6-31G(d,p) IRC=Calcfc
```

Requires: A valid transition state (confirmed by frequency).

## Polarizability and Hyperpolarizability

```gjf
#p B3LYP/6-311++G(d,p) Polar=Polarizability
```

## NMR Chemical Shifts

```gjf
#p B3LYP/6-311+G(d,p) NMR
```

Requires: GIAO (Gauge-Including Atomic Orbitals) is automatic in Gaussian.

## Excited States (TD-DFT)

Time-dependent DFT for excited states:

```gjf
#p B3LYP/6-311+G(d,p) TD(NStates=10)
```

Requests 10 excited states.

## Composite Methods

### CBS-QB3

```gjf
#CBS-QB3
```

Performs a series of calculations at increasing theory level:
1. B3LYP/6-311G(d,p) geometry + frequency
2. MP2/6-311G(d,p) energy
3. CCSD(T)/6-311+G(d,p) single point

### G3

```gjf
#G3
```

Higher accuracy than CBS-QB3.

## Job Sequencing

For complex workflows, split into stages:

1. **Stage 1:** Geometry optimization
2. **Stage 2:** Frequency at optimized geometry
3. **Stage 3:** High-level single point

```gjf
%Chk=stage1.chk
#p B3LYP/6-31G(d,p) Opt

Stage 1: Optimize
-1 1
...geometry...

---
%Chk=stage1.chk
%OldChk=stage1.chk
#p B3LYP/6-31G(d,p) Freq

Stage 2: Frequencies
...
```
