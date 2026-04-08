---
name: abrsa
description: Robust antibody numbering and CDR (complementarity determining region) delimitation tool. Use when:(1) numbering antibody sequences (heavy chain, light chain, nanobodies), (2) delimiting CDR regions, (3) converting between IMGT/Kabat/Chothia numbering schemes. Command-line tool for Linux x86-64. 
metadata: 
    skill-author: Co-authored by Yang Cao Lab and MindSpore Science Team
---

# AbRSA

Antibody Region-Specific Alignment for numbering and CDR delimitation.


## Installation

AbRSA is a command-line application designed for Linux-x86 (64-bit Intel/AMD) systems. The latest version, updated in May 2022, can be downloaded from the official website: http://cao.labshare.cn/AbRSA/download.html

## Basic Usage

```bash
AbRSA -i [query_antibody.fas]
```

## Options

| Option | Description |
|--------|-------------|
| `-i` | Input antibody amino acid FASTA file (required) |
| `-o` | Output numbering file |
| `-c` | Use Chothia scheme (default) |
| `-k` | Use Kabat scheme |
| `-g` | Use IMGT scheme |
| `-z 70` | Similarity cutoff (default: 70) |
| `-h` | Show help |

## Numbering Schemes

- **Chothia** (default): Numbering based on CDRs as in Chothia et al.
- **Kabat**: Based on Kabat et al. numbering
- **IMGT**: IMGT numbering scheme

## Output Format

```
>3OAU-H
#similarity 88 %

H_FR1 : EVQLVESGGGLVKAGGSLRLSCGVS
H_CDR1: NFRISAH
H_FR2 : TMNWVRRVPGGGLEWVASI
H_CDR2: STSSTY
H_FR3 : RDYADAVKGRFTVSRDDLEDFVYLQMHKMRVEDTAIYYCAR
H_CDR3: KGSDRLSDNDPFDA
H_FR4 : WGPGTVVTVSPA
-_EXT : STKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKS
```

- `H_` prefix = Heavy chain, `L_` prefix = Light chain
- `FR1-4` = Framework regions
- `CDR1-3` = Complementarity determining regions
- `-_EXT` = Constant region / extension

## Chain Detection

AbRSA automatically detects:
- Heavy chain sequences
- Light chain sequences
- Single-chain antibodies (nanobodies)

## Examples

**Chothia numbering (default):**
```bash
AbRSA -i ab.fas -c -o ab_numbering.txt
```

**IMGT numbering:**
```bash
AbRSA -i ab.fas -g -o ab_numbering.txt
```

**Kabat numbering:**
```bash
AbRSA -i ab.fas -k -o ab_numbering.txt
```
