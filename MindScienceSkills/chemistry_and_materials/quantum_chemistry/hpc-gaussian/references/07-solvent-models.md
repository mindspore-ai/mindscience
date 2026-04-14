# Solvent Models

## When to Use Solvent

Solvent effects are important for:
- Properties in solution (pKa, redox potentials)
- Reactions in solution
- Spectroscopy (UV-Vis, NMR shifts)
- Molecular recognition

## PCM (Polarizable Continuum Model)

The most common implicit solvent model in Gaussian.

### Basic PCM

```gjf
#p B3LYP/6-311G(d,p) SCRF=PCM

Water solvent

0 1
O   0.0  0.0  0.0
H   0.96  0.0  0.0
H  -0.24  0.93  0.0
```

### Specify Solvent

```gjf
#p B3LYP/6-311G(d,p) SCRF=(PCM,Solvent=Acetonitrile)
```

Common solvents:

| Solvent | Keyword |
|---------|---------|
| Water | `Water` |
| Methanol | `Methanol` |
| Acetonitrile | `Acetonitrile` |
| Chloroform | `Chloroform` |
| Dichloromethane | `Dichloromethane` |
| DMSO | `DMSO` |
| Benzene | `Benzene` |
| Toluene | `Toluene` |

## SMD (Solvation Model Density)

SMD is generally more accurate than PCM for:
- Thermochemistry in solution
- pKa predictions
- Reduction potentials

```gjf
#p B3LYP/6-311G(d,p) SCRF=(SMD,Solvent=Water)
```

SMD parameters are derived from DFT electron densities.

## CPCM (Conductor-like PCM)

Variant of PCM with different parameterization:

```gjf
#p B3LYP/6-311G(d,p) SCRF=(CPCM,Solvent=Water)
```

## Comparing Solvent Models

| Model | Accuracy | Speed | Notes |
|-------|----------|-------|-------|
| PCM | Good | Fast | Most common |
| SMD | Better | Similar to PCM | Better for thermochemistry |
| CPCM | Good | Fast | Related to PCM |
| COSMO | Good | Fast | Similar to PCM |

## Explicit-Implicit Combinations

For specific solute-solvent interactions:

```gjf
#p B3LYP/6-311G(d,p) SCRF=(PCM,Solvent=Water) guess=read

Solvent with explicit water

0 1
...solute with explicit water molecules...

---
Forces (Solvent=Water)  # Add bulk solvent forces
```

This adds explicit solvent molecules around the solute.

## Non-Equilibrium Solvation

For excited states (vertical transitions):

```gjf
#p B3LYP/6-311G(d,p) TD=(NStates=5) SCRF=(PCM,Solvent=Water,NonEquilibrium=Save)
```

This calculates the vertical excitation with the ground state solvent
response frozen.

## Temperature-Dependent Solvent Effects

```gjf
#p B3LYP/6-311G(d,p) SCRF=(PCM,Solvent=Water,Temperature=298)
```

Temperature affects dielectric constant and solvent response.

## Choosing a Solvent Model

| Use Case | Recommended Model |
|----------|-------------------|
| General solvation | PCM or SMD |
| Thermochemistry | SMD |
| pKa, redox potentials | SMD or PCM |
| UV-Vis in solution | PCM or CPCM |
| Gas phase vs solution | Both with same model |
| Enzyme (QM/MM) | Explicit QM/MM preferred |

## Common Solvent Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `Solvent not supported` | Wrong solvent name | Use correct keyword |
| `PCM not initialized` | Bad geometry | Optimize in gas phase first |
| `Solvent reaction field failed` | Numerical issues | Use smaller basis set first |

## Solvent for Different Calculations

| Calculation | With Solvent? | Notes |
|------------|--------------|-------|
| Geometry optimization | Yes | Consistent with experiment |
| Frequency | Yes | Include ZPE in solvent |
| Single point energy | Yes | Solvent effect on energy |
| NMR | Yes | Solvent shifts are important |
| Gas phase comparison | No | Compare apples to apples |
