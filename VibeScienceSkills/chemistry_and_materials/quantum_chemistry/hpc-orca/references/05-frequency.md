# Frequency Calculations and Thermodynamics

## Basic Frequency Calculation

```
! B3LYP def2-SVP FREQ
```

## Frequency After Optimization

```
! B3LYP def2-SVP OPT FREQ
```

## Raman Spectroscopy

```
! B3LYP def2-SVP FREQ RAMAN
```

## Output Contents

- Vibrational frequencies (cm⁻¹)
- IR intensities
- Raman activities
- Thermodynamic properties
  - Zero-point energy (ZPE)
  - Enthalpy (H)
  - Entropy (S)
  - Gibbs free energy (G)

## Thermodynamic Corrections

```
%freq
  Temp 298.15    # Temperature (K)
  ScaleFactor 0.98 # Frequency scaling factor
end
```

## Stationary Point Verification

| Frequency Result | Stationary Point Type |
|------------------|------------------------|
| All positive frequencies | Minimum (stable structure) |
| 1 imaginary frequency | Transition state (first-order saddle point) |
| Multiple imaginary frequencies | Higher-order saddle point |

## Isotope Effects

```
%freq
  Isotopes
    { 1 2 }  # Atom 1 uses isotope 2
  end
end
```

## Frequency Scaling Factors

| Functional/Basis Set | Scaling Factor |
|---------------------|----------------|
| B3LYP/6-31G* | 0.961 |
| B3LYP/def2-TZVP | 0.987 |
| wB97X-D/def2-TZVP | 0.985 |