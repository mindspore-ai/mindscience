# Solvent Effects

## Implicit Solvent Models

### CPCM

```
! B3LYP def2-SVP CPCM(water)
```

### SMD

```
! B3LYP def2-SVP SMD(water)
```

## Common Solvents

| Solvent | Keyword |
|---------|---------|
| Water | water |
| Methanol | methanol |
| Ethanol | ethanol |
| Acetonitrile | acetonitrile |
| Dichloromethane | dichloromethane |
| Toluene | toluene |
| Benzene | benzene |
| Chloroform | chloroform |
| DMSO | dmso |
| THF | thf |

## Explicit Solvent

```
* xyz 0 1
[solute molecule]
[explicit solvent molecules]
*
```

Combined with implicit solvent:
```
! B3LYP def2-SVP CPCM(water)
```

## Solvent Parameters

```
%cpcm
  epsilon 78.4      # Dielectric constant
  refrac 1.33        # Refractive index
end
```

## pKa Calculation

Using thermodynamic cycle:
1. Gas-phase deprotonation energy
2. Solvation free energy
3. Proton solvation energy