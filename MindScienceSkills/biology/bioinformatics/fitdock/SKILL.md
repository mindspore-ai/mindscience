---
name: fitdock
description: Protein-ligand docking tool with template-based alignment for improved accuracy. Use when:(1) performing protein-ligand docking with template guidance, (2) aligning ligand structures, (3) checking non-protein components in PDB files. Achieves 40-60% improvement in docking success rate when template similarity > 0.5. Command-line tool for Linux/Windows x86-64.
metadata: 
    skill-author: Co-authored by Yang Cao Lab and MindSpore Science Team
---

# FitDock

Template-based protein-ligand docking with hierarchical multi-feature alignment.


## Installation

FitDock is an enhanced template fitting-based protein-ligand docking software. The associated methodology has been published in Briefings in Bioinformatics(2022).

**Obtaining the Software:**

To download FitDock, please visit its official registration page, complete the information form, and you will then be able to acquire the software.
Registration and Download Page: http://cao.labshare.cn/fitdock/php/register.php


## Basic Usage

**Template-based docking:**
```bash
FitDock -Tprot [template_protein.pdb] -Tlig [template_ligand.mol2] -Qprot [query_protein.pdb] -Qlig [query_ligand.mol2] -o [output.mol2]
```

## Required Parameters

| Option | Description |
|--------|-------------|
| `-Tprot` | Template protein structure (PDB) |
| `-Tlig` | Template ligand structure (MOL2) |
| `-Qprot` | Query protein structure (PDB) |
| `-Qlig` | Query ligand structure (MOL2) |
| `-o` | Output final query ligand structure (MOL2) |

## Optional Output Parameters

| Option | Description |
|--------|-------------|
| `-os` | Output aligned query ligand on template (MOL2) |
| `-ot` | Output template ligand after alignment (MOL2) |
| `-oT` | Output template protein after alignment (PDB) |

## Optional Parameters

| Option | Description |
|--------|-------------|
| `-align_only` | Align query ligand to template without protein docking |
| `-am 0/1` | Amide bond rotatable: 0=non-rotatable, 1=rotatable (default) |
| `-keep_het` | Comma-separated HETATM IDs to retain |
| `-acc 0/1` | Accuracy mode: 0=fast (default), 1=more accurate but slower |
| `-h` | Print help |

## Ligand Structure Alignment Only

```bash
FitDock -Tlig [template.mol2] -Qlig [query.mol2] -align_only -o [output.mol2]
```

## Check Non-Protein Components

```bash
FitDock -Qprot [protein.pdb] -check_het
```

## Output Interpretation

```
Ligand Similarity(Q): 0.919   # PC-Score based on query ligand
Ligand Similarity(T): 0.999   # PC-Score based on template ligand
Pocket Similarity:    1       # Pocket region sequence identity
Pocket RMSD:         0.0945  # Pocket region structure RMSD
Binding Score before EM: -6.86
Binding Score after EM: -7.6
Comprehensive Score: 0.951   # Combined docking + PC-score
```

Higher Comprehensive Score = better template match.

## Examples

**Standard template docking:**
```bash
cd examples
../bin/FitDock -Tprot 5iwg_protein_template.pdb -Tlig 5iwg_ligand_template.mol2 \
    -Qprot 5ix0_protein_query.pdb -Qlig 5ix0_ligand_query.mol2 \
    -ot ot.mol2 -os os.mol2 -o o.mol2
```

**Accuracy mode (slower but more accurate):**
```bash
FitDock -Tprot template.pdb -Tlig template.mol2 -Qprot query.pdb -Qlig query.mol2 \
    -o output.mol2 -acc 1
```

**Ligand-only alignment:**
```bash
FitDock -Tlig template.mol2 -Qlig query.mol2 -align_only -o aligned.mol2
```

## Performance Notes

- FitDock is 10x faster than popular docking methods when template similarity > 0.5
- 40-60% improvement in docking success rate vs. conventional methods
- Works on both Linux and Windows x86-64 systems



## Please Cite

Xiaocong Yang, Yang Liu, Jianhong Gan, Zhi-Xiong Xiao, Yang Cao. FitDock: improved protein-ligand docking with template fitting. *Briefings in Bioinformatics*. 23(3), 2022, https://doi.org/10.1093/bib/bbac087