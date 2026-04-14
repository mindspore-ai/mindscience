---
name: abalign
description: High-throughput, high-accuracy multiple sequence alignment (MSA) tool for antibody/B-cell receptor sequences. Use when:(1) aligning antibody protein or DNA sequences, (2) numbering antibodies with IMGT/Kabat/Chothia schemes, (3) delimiting CDR regions, (4) analyzing V/J gene usage and clonotypes, (5) calculating residue frequency and region enrichment.
metadata: 
    skill-author: Co-authored by Yang Cao Lab and MindSpore Science Team
---

# Abalign

High-throughput MSA tool for antibody immune repertoires with built-in numbering, CDR delimitation, and gene analysis.


## Installation

Abalign is a tool for B-cell receptor multiple sequence alignment, offering both a web server and multi-platform desktop clients.

**Obtaining the Software:**

- To download the Abalign software, you need to visit its official download page and complete a free registration.
Download Page: http://47.108.188.197/abalign/download

- Registration Requirement: Simple registration information is required before download. The process is quick, and your information will be kept confidential.

## Basic Usage

**Protein sequence input:**
```bash
Abalign -i [input.fas] -ah [heavy_output.fas] -al [light_output.fas] [options]
```

**DNA sequence input (with translation):**
```bash
Abalign -n [input.fas] -p [translated.fas] -ah [output.fas] [options]
```

## Core Options

| Option | Description |
|--------|-------------|
| `-ah/-al` | Output heavy/light chain MSA (FASTA) |
| `-o` | Output antibody numbering file |
| `-r` | Mark different regions with `*` in MSA |
| `-mg` | Remove redundant sequences (dedup) |
| `-bd` | Calculate sequence abundance |
| `-s` | Silent mode for high-throughput |
| `-t` | Thread count (default: 1) |

## Numbering Schemes

| Option | Scheme |
|--------|--------|
| `-g` | IMGT (default) |
| `-c` | Chothia |
| `-k` | Kabat |

## Gene & Clonotype Analysis

```bash
Abalign -i input.fas -ah out.fas -sp HS. -v genes.txt -vct 0
```

- `-sp HS.` - Species: HS (human), MM (mouse), BT (bovine), etc. **Must end with dot.**
- `-v` - Output V gene identification file
- `-vct 0` - Output clonotypes (same V/J + identical CDR3)
- `-vct 1` - Output clonotypes with high CDR3 consistency
- `-vn N` - Output top-N V genes (default: 3)

## Filtering & Parameters

| Option | Description |
|--------|-------------|
| `-z 70` | Similarity threshold (lower = more sequences kept, default: 70) |
| `-l 70` | Minimum protein length after translation (DNA mode) |
| `-lfs 2` | Length filter: 0=off, 1=normal, 2=strict, 3=loose |
| `-x` | Remove insertions from MSA |
| `-ra 0` | Remove low-abundance sequences (threshold) |

## Region Enrichment & Length Calculation

```bash
Abalign -i input.fas -re -rg 8 -lc -lg 6
```

- `-re` - Output region enrichment file
- `-rg 8` - Region for enrichment: 1=FR1, 2=CDR1, 3=FR2, 4=CDR2, 5=FR3, 6=CDR3, 7=FR4, 8=all
- `-lc` - Output length calculation file
- `-lg 6` - Region for length calculation (default: 6=CDR3)

## Residue Frequency

```bash
Abalign -mf [msa.fas] -in [numbering.txt]
```

## Examples

**MSA with dedup and abundance:**
```bash
Abalign -i protein.fas -ah msa_heavy.fas -r -s -t 6 -mg -bd
```

**Human antibody analysis with clonotypes:**
```bash
Abalign -i protein.fas -ah msa.fas -r -s -t 6 -sp HS. -v genes.txt -vct 0
```

**Relaxed filtering for diverse sequences:**
```bash
Abalign -i protein.fas -ah msa.fas -r -s -t 6 -z 50
```

**DNA input with translation:**
```bash
Abalign -n dna.fas -p translated.fas -ah msa.fas -r -s -t 6 -z 60 -l 50
```

## Notes

1. Always end species codes with `.` (e.g., `-sp HS.`, `-sp HS,MM.`)
2. Use `-s` for large datasets to suppress screen output
3. Use `-ah` for heavy chain, `-al` for light chain

## Cite

F Zong#, C Long#, W Hu#, S Chen, W Dai, ZX Xiao, Y Cao*. Abalign: a comprehensive multiple sequence alignment platform for B-cell receptor immune repertoires. *Nucleic Acids Research*. 51 (W1), W17-W24, 2023.