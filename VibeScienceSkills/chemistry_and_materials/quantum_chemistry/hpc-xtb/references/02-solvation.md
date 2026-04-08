# XTB Solvation Models

## GBSA Solvation Model

```bash
# Water solvent
xtb molecule.xyz --sp --gbsa water

# Organic solvents
xtb molecule.xyz --sp --gbsa acetone
xtb molecule.xyz --sp --gbsa ethanol
```

## Supported Solvents

| Solvent | Name |
|---------|------|
| Water | water |
| Acetone | acetone |
| Acetonitrile | acetonitrile |
| DMF | dmf |
| DMSO | dmso |
| Ethanol | ethanol |
| Methanol | methanol |
| THF | thf |
| Toluene | toluene |
| Chloroform | chloroform |
| Dichloromethane | dichloromethane |
| n-Hexane | hexane |

## Solvent Parameters

```bash
# Custom solvent parameters
xtb molecule.xyz --sp --gbsa water --alpb water
```