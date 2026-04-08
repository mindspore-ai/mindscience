# DFT Methods and Functional Selection

## Functional Classification

### LDA (Local Density Approximation)
- SVWN, VWN5
- Suitable for: uniform electron gas approximation

### GGA (Generalized Gradient Approximation)
- PBE, BP86, BLYP
- Suitable for: general chemical problems

### Meta-GGA
- TPSS, M06-L
- Suitable for: systems requiring kinetic energy density

### Hybrid Functionals
- B3LYP (20% HF)
- PBE0 (25% HF)
- Suitable for: most organic molecules

### Range-Separated Functionals
- wB97X-D, CAM-B3LYP
- Suitable for: charge transfer, excited states

### Double Hybrid Functionals
- B2PLYP, DSD-PBEP86
- Suitable for: high-precision energy calculations

## Functional Recommendations

| Application | Recommended Functional | Reason |
|-------------|------------------------|--------|
| Routine calculations | B3LYP | Good balance |
| Thermochemistry | M06-2X | High non-locality |
| Excited states | wB97X-D | Long-range correction |
| Weak interactions | B97-D3 | Dispersion correction |
| Metal systems | TPSS | Meta-GGA |
| High precision | DLPNO-CCSD(T) | Wavefunction method |

## Dispersion Correction

```
! B3LYP D3BJ    # D3 with Becke-Johnson damping
! B3LYP D4      # D4 correction
! wB97X-D       # Built-in dispersion correction
```

## RI Acceleration

```
! B3LYP RIJCOSX  # RI-J + COSX integration
! B3LYP RI-J     # RI-J acceleration only
```

## Wavefunction Methods

```
! MP2 def2-TZVP           # MP2
! DLPNO-CCSD(T) def2-TZVP # High precision
! SCS-MP2                 # Spin-component scaled MP2
```