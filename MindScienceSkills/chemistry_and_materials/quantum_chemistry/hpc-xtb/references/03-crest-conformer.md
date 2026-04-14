# CREST Conformer Search

## Basic Usage

```bash
crest molecule.xyz -gfn2
```

## Common Options

| Option | Description |
|--------|-------------|
| -gfn2 | Use GFN2-xTB |
| -gfn1 | Use GFN1-xTB |
| -T N | Use N threads |
| -solvent NAME | Solvent model |
| -ewin N | Energy window (kcal/mol) |
| -rthr N | RMSD threshold |

## Examples

```bash
# Conformer search in water solvent
crest molecule.xyz -gfn2 -solvent water

# Quick search (GFN1-xTB)
crest molecule.xyz -gfn1 -T 8

# Large energy window
crest molecule.xyz -gfn2 -ewin 10
```

## Output Files

| File | Description |
|------|-------------|
| crest_conformers.xyz | All conformers |
| crest.energies | Conformer energies |
| crest_best.xyz | Lowest energy conformer |
| crest_rotamers.xyz | Rotamers |

## Workflow

1. Generate initial conformers
2. Pre-optimization (GFN-FF)
3. Fine optimization (GFN-xTB)
4. Deduplication and ranking
5. Output results