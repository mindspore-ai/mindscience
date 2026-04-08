# XTB Methods Overview

## GFN-xTB Method Family

| Method | Features | Use Cases |
|--------|----------|-----------|
| GFN2-xTB | Latest version, highest accuracy | Default choice |
| GFN1-xTB | Faster | Large systems |
| GFN-FF | Force field method | Very large systems |
| GFN0-xTB | Simplest | Quick preview |

## Usage

```bash
# GFN2-xTB (default)
xtb molecule.xyz --sp

# GFN1-xTB
xtb molecule.xyz --sp --gfn 1

# GFN-FF
xtb molecule.xyz --sp --gfnff
```

## Calculation Types

| Type | Command | Description |
|------|---------|-------------|
| Single point | --sp | Energy calculation |
| Geometry optimization | --opt | Structure optimization |
| Frequency | --hess | Vibrational analysis |
| MD | --md | Molecular dynamics |
| Transition state | --gfnff-ts | TS search |

## Accuracy and Speed

| Method | Relative Speed | Relative Accuracy |
|--------|----------------|-------------------|
| GFN2-xTB | 1x | Highest |
| GFN1-xTB | 2x | Medium |
| GFN-FF | 10x | Lower |

## Applicable Systems

- Organic molecules
- Main group compounds
- Transition metal complexes
- Biomolecules
- Solid surfaces