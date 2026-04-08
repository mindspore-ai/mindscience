# Basis Set Selection Guide

## def2 Series (Recommended)

| Basis Set | Level | Application |
|-----------|-------|-------------|
| def2-SVP | Double-zeta + polarization | Geometry optimization, prescanning |
| def2-TZVP | Triple-zeta + polarization | Single point energy, spectroscopy |
| def2-QZVP | Quadruple-zeta + polarization | High-precision calculations |
| def2-TZVPP | Triple-zeta + double polarization | Higher precision |

## Pople Series

| Basis Set | Description |
|-----------|-------------|
| 6-31G | Basic double-zeta |
| 6-31G* | + polarization functions |
| 6-31G** | + hydrogen polarization |
| 6-311G** | Triple-zeta + polarization |

## Correlation-Consistent Basis Sets

| Basis Set | Application |
|-----------|-------------|
| cc-pVDZ | Double-zeta |
| cc-pVTZ | Triple-zeta |
| cc-pVQZ | Quadruple-zeta |
| aug-cc-pVTZ | + diffuse functions |

## Effective Core Potential (ECP)

```
! def2-TZVP def2-ECP  # Automatically use ECP
```

Suitable for: heavy elements (transition metals, lanthanides, actinides)

## Basis Set Recommendations

| System | Recommended Basis Set |
|--------|----------------------|
| Small organic molecules | def2-TZVP |
| Large molecules | def2-SVP |
| Anions | aug-cc-pVTZ |
| Transition metals | def2-TZVP + ECP |
| High precision | def2-QZVPP |