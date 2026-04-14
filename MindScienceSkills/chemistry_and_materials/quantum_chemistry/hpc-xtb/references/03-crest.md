# CREST Conformer Search

## Basic Usage

```bash
# GFN2-xTB conformer search
crest molecule.xyz --gfn2

# Using solvent model
crest molecule.xyz --gfn2 --gbsa water

# Specify parallel threads
crest molecule.xyz --gfn2 -T 8
```

## Search Options

| Option | Description |
|--------|-------------|
| --gfn2 | Use GFN2-xTB |
| --gfn1 | Use GFN1-xTB |
| --gfnff | Use GFN-FF |
| --gbsa | Solvent model |
| -T | Number of parallel threads |
| --etol | Energy convergence |
| --gcinp | Input conformers |

## Output Files

| File | Description |
|------|-------------|
| crest_conformers.xyz | All conformers |
| crest_best.xyz | Lowest energy conformer |
| crest.energies | Energy list |
| gfnff_topo | Topology file |

## Filter Options

```bash
# Energy window
crest molecule.xyz --gfn2 --ewin 10.0

# Maximum number of conformers
crest molecule.xyz --gfn2 --maxconf 100
```