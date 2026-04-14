# Method and Basis Set Selection

## Method Hierarchy

```
Low accuracy ──────────────────────────────────── High accuracy
   │
   ├── Hartree-Fock (HF)
   │       │
   │       └── DFT (B3LYP, ωB97X-D, etc.)
   │               │
   │               ├── MP2 (Møller-Plesset)
   │               │       │
   │               │       └── CCSD(T) (gold standard)
   │               │
   │               └── DFT with larger basis
   │
   └── Higher accuracy with larger basis
```

## Hartree-Fock (HF)

- Foundation for all methods
- Includes exchange but NO correlation
- Fastest, scales as N⁴

**Use for:** Testing, initial guesses, large systems where correlation is minor.

## Density Functional Theory (DFT)

DFT includes both exchange and correlation through functionals.

### Popular Functionals

| Functional | Type | Notes |
|-----------|------|-------|
| B3LYP | Hybrid | Most widely used, good general-purpose |
| ωB97X-D | Range-separated hybrid | Good for non-covalent |
| M06-2X | Hybrid | Good for thermochemistry |
| PBEPBE | GGA | Standard, not best for thermochemistry |
| WB97X-V | Dispersion-corrected | Excellent for complexes |

### Functional Selection Guide

| Use Case | Recommended Functional |
|----------|---------------------|
| General purpose | B3LYP |
| Non-covalent interactions | ωB97X-D, WB97X-V |
| Thermochemistry | M06-2X, WB97X-D |
| Ground state organic | B3LYP, M06-2X |
| Excited states | TD-DFT with range-separated |
| Charge transfer | Range-separated (ωB97X-D) |

## Wavefunction Methods

### MP2 (Møller-Plesset 2nd order)

- Includes electron correlation
- Scales as N⁵
- Good for medium systems (~50 atoms)

```gjf
#MP2/6-311G(d,p)
```

### CCSD(T) (Coupled Cluster)

- "Gold standard" for single reference
- Scales as N⁷ (CCSD(T))
- Use for final energies on small systems

```gjf
#CCSD(T)/cc-pVDZ
```

## Basis Sets

### Pople Basis Sets (6-31G family)

| Basis | Description |
|-------|-------------|
| `6-31G` | Split-valence, double-zeta |
| `6-311G` | Triple-zeta valence |
| `6-31G(d,p)` | + d on heavy atoms, p on H |
| `6-311G(d,p)` | Triple-zeta + polarization |

### Correlation-Consistent (Dunning)

| Basis | Description |
|-------|-------------|
| `cc-pVDZ` | Double-zeta |
| `cc-pVTZ` | Triple-zeta |
| `cc-pVQZ` | Quadruple-zeta |
| `aug-cc-pVDZ` | + diffuse functions |

### Selection Guide

| System Size | Recommended Basis |
|------------|------------------|
| < 20 atoms | 6-311G(d,p) or cc-pVTZ |
| 20-50 atoms | 6-31G(d,p) or cc-pVDZ |
| 50-100 atoms | 6-31G or cc-pVDZ |
| > 100 atoms | STO-3G or 3-21G (screening) |

## Composite Methods

### CBS-QB3

Complete basis set extrapolation for accurate thermochemistry:

```gjf
#CBS-QB3
```

Good for: Accurate energies, reaction barriers, thermochemistry.

## Effective Core Potentials (ECP)

For heavy elements (Z > 36):

```gjf
#B3LYP/LANL2DZ
```

Or use:
- `Def2TZVPP` — Karlsruhe def2 basis with ECP
- `SDD` — Stuttgart-Dresden ECPs

## Method + Basis Recommendations

| Purpose | Method | Basis | Notes |
|---------|--------|-------|-------|
| Geometry optimization | B3LYP | 6-31G(d,p) | Good accuracy |
| Frequency (ZPE, entropy) | B3LYP | 6-31G(d,p) | Match opt level |
| Single point energy | CCSD(T) | cc-pVTZ | High accuracy |
| Non-covalent | ωB97X-D | aug-cc-pVDZ | Dispersion important |
| TS/barrier | M06-2X | 6-311+G(d,p) | Good for kinetics |

## Open Shell Systems

For radicals, use unrestricted methods:

```gjf
#UB3LYP/6-311G(d,p)
```

For restricted open-shell:

```gjf
#ROB3LYP/6-311G(d,p)
```

## Common Mistakes

| Mistake | Problem | Fix |
|---------|---------|-----|
| Using HF for geometry | Ignores correlation | Use DFT or MP2 |
| Mixing method and basis | Wrong level comparison | Consistent throughout |
| Too small basis | Basis set superposition error | Use polarization functions |
| Using DFT for dispersion | No dispersion by default | Use dispersion-corrected DFT |
