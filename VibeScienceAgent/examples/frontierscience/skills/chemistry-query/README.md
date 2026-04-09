# Chemistry Query Agent 🧪

## OpenClaw Skill for Chem Informatics

PubChem data + RDKit analysis/screening/viz/reactions. Agent-ready.

## Install

```
clawhub install chemistry-query
```

Or git clone + OpenClaw skills dir.

## Demos

**1. Aspirin analysis:**
```
exec python skills/chemistry-query/scripts/query_pubchem.py --compound aspirin --type info
{"CID":2244,"MolecularFormula":"C9H8O4","MolecularWeight":180.16}
exec python skills/chemistry-query/scripts/rdkit_mol.py --smiles "CC(=O)Oc1ccccc1C(=O)O" --action props
{"logp":1.31,"tpsa":63.6}
exec python skills/chemistry-query/scripts/rdkit_mol.py --smiles "..." --action draw --format svg > aspirin.svg
```

**GIF:** [screen input → viz/table](GIF_PLACEHOLDER)

**2. Virtual screen:** Sim + ADMET filter.

**3. Synth plan:** Reaction templates + lit.

## Features

- PubChem: Info/structure/similar/lit/synthesis
- RDKit: Props/viz (PNG/SVG/XYZ)/FP/sim/substruct/react
- ADMET rules (Ro5/Veber/PAINS)
- Batch CSV
- PubMed refs

ClawHub soon. Stars/contribs welcome!

Cheminem