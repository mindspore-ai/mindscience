---
name: tooluniverse-admet-prediction
description: Comprehensive ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) profiling of drug candidates using ADMETAI predictions, SwissADME drug-likeness, PubChemTox experimental toxicity, ChEMBL clinical data, and PubChem properties. Generates a structured ADMET scorecard with pass/fail verdicts per category. Use when asked about drug-likeness, ADMET properties, bioavailability, toxicity prediction, BBB penetration, CYP interactions, pharmacokinetic profiling, Lipinski rule of five, or ADME/PK assessment of a compound.

license: MIT License
metadata:
    skill-author: ToolUniverse (https://github.com/mims-harvard/TxAgent)
---

# ADMET Prediction & Drug Candidate Profiling

Comprehensive pharmacokinetic and toxicity profiling integrating AI-based ADMET predictions, rule-based drug-likeness filters, and experimental benchmarks from curated databases.

## When to Use This Skill

**Triggers**:
- "What are the ADMET properties of [compound]?"
- "Is [drug] likely to cross the blood-brain barrier?"
- "Predict the toxicity of this SMILES: ..."
- "Does [compound] violate Lipinski's rule of five?"
- "Assess the drug-likeness of [molecule]"
- "What are the CYP interactions for [drug]?"
- "Pharmacokinetic profile of [compound]"
- "Is [compound] orally bioavailable?"
- "What is the LD50 / hERG liability of [molecule]?"

**Input**: Drug name (e.g., "ibuprofen") OR SMILES string (e.g., "CC(C)Cc1ccc(cc1)C(C)C(=O)O")

---

## KEY PRINCIPLES

1. **Resolve identity first** - Always convert drug name to SMILES before calling ADMETAI tools
2. **ADMETAI tools require `tooluniverse[ml]`** - If import fails, skip to SwissADME/PubChemTox fallbacks
3. **All ADMETAI tools take `smiles: list[str]`** - Always wrap in a list, even for one compound
4. **SwissADME takes `smiles: str`** - Single string, NOT a list (SOAP-style with `operation` param)
5. **PubChemTox tools accept `cid` or `compound_name`** - Use CID when available for reliability
6. **Evidence grading mandatory** - Predictions (T3), experimental data (T2), regulatory (T1)
7. **Scorecard output** - Every analysis must end with a pass/warn/fail scorecard
8. **Explain significance** - State WHY each property matters for drug development

---

## Evidence Grading

| Tier | Label | Source |
|------|-------|--------|
| **T1** | Regulatory/Clinical | FDA labels, ChEMBL max clinical phase |
| **T2** | Experimental | PubChemTox LD50/LC50, in vitro AMES, animal studies |
| **T3** | Computational | ADMETAI predictions, SwissADME calculations |
| **T4** | Annotation | Database cross-references, text-mined |

---

## Tool Reference (Verified Parameters)

### Compound Resolution
| Tool | Parameters | Returns |
|------|-----------|---------|
| `PubChem_get_CID_by_compound_name` | `name: str` | `{IdentifierList: {CID: [int]}}` |
| `PubChem_get_CID_by_SMILES` | `smiles: str` | `{IdentifierList: {CID: [int]}}` |
| `PubChem_get_compound_properties_by_CID` | `cid: int` | `{CID, MolecularWeight, ConnectivitySMILES, IUPACName}` |

### ADMETAI Predictions (all take `smiles: list[str]`)
| Tool | Predicts | Key Outputs |
|------|----------|-------------|
| `ADMETAI_predict_physicochemical_properties` | MW, logP, PSA, HBD, HBA | Lipinski descriptors |
| `ADMETAI_predict_solubility_lipophilicity_hydration` | Aqueous solubility | LogS, hydration free energy |
| `ADMETAI_predict_BBB_penetrance` | Blood-brain barrier crossing | BBB+/BBB- classification |
| `ADMETAI_predict_bioavailability` | Oral bioavailability | F20%/F30% classification |
| `ADMETAI_predict_CYP_interactions` | CYP450 substrate/inhibitor | CYP1A2, 2C9, 2C19, 2D6, 3A4 |
| `ADMETAI_predict_clearance_distribution` | VDss, clearance, PPB | PK parameters |
| `ADMETAI_predict_toxicity` | AMES, DILI, hERG, LD50, ClinTox | Safety flags |
| `ADMETAI_predict_nuclear_receptor_activity` | NR-AR, NR-AhR, NR-ER, etc. | Endocrine disruption |
| `ADMETAI_predict_stress_response` | p53, HSE, MMP, ATAD5 | Genotoxicity stress |

### SwissADME (SOAP-style: requires `operation` param)
| Tool | Parameters | Returns |
|------|-----------|---------|
| `SwissADME_calculate_adme` | `operation: "calculate_adme"`, `smiles: str` | 49 ADME properties |
| `SwissADME_check_druglikeness` | `operation: "check_druglikeness"`, `smiles: str` | Lipinski/Veber/Ghose/Egan/Muegge |

### PubChemTox (all take `cid: int` OR `compound_name: str`)
| Tool | Returns |
|------|---------|
| `PubChemTox_get_toxicity_values` | LD50, LC50 from animal studies |
| `PubChemTox_get_ghs_classification` | GHS hazard codes and statements |
| `PubChemTox_get_acute_effects` | Acute toxicity by route/species |
| `PubChemTox_get_carcinogen_classification` | IARC/NTP/EPA carcinogen class |
| `PubChemTox_get_target_organs` | Organ-specific toxicity |
| `PubChemTox_get_toxicity_summary` | Overall toxicity summary |

### Clinical Benchmarks
| Tool | Parameters | Returns |
|------|-----------|---------|
| `ChEMBL_get_molecule` | `chembl_id: str` | Max phase, Ro5 violations, molecular properties |

---

## Workflow: 5-Phase ADMET Profiling

```
User Query (drug name or SMILES)
|
+-- PHASE 1: Compound Identity Resolution
|   PubChem name->CID->SMILES, or validate input SMILES
|
+-- PHASE 2: Physicochemical & Drug-Likeness
|   ADMETAI physicochemical + SwissADME druglikeness -> Lipinski/Veber
|
+-- PHASE 3: ADME Predictions
|   BBB, bioavailability, CYP interactions, clearance, solubility
|
+-- PHASE 4: Toxicity Assessment
|   ADMETAI tox + PubChemTox experimental + nuclear receptor + stress
|
+-- PHASE 5: Scorecard & Clinical Context
|   ChEMBL max phase, aggregate pass/warn/fail, final recommendation
```

---

### PHASE 1: Compound Identity Resolution

**Goal**: Obtain SMILES, PubChem CID, and basic identifiers for the query compound.

**Steps**:

1. **If input is a drug name**:
   - Call `PubChem_get_CID_by_compound_name(name=<drug_name>)` to get CID
   - Call `PubChem_get_compound_properties_by_CID(cid=<CID>)` to get SMILES and MW
   - Extract `ConnectivitySMILES` from the response (NOT `CanonicalSMILES`)

2. **If input is a SMILES string**:
   - Call `PubChem_get_CID_by_SMILES(smiles=<SMILES>)` to get CID
   - Call `PubChem_get_compound_properties_by_CID(cid=<CID>)` for compound name and MW
   - Use the input SMILES for all subsequent ADMETAI calls

3. **Record**:
   - Compound name, CID, SMILES, molecular formula, molecular weight, IUPAC name
   - If CID lookup fails, proceed with SMILES only (ADMETAI does not need CID)

**Why this matters**: ADMETAI tools require SMILES input. PubChemTox tools work best with CID. Resolving both ensures all downstream tools can be called. PubChem is the authoritative source for SMILES canonicalization.

**Fallback**: If PubChem has no entry, the user must provide SMILES directly. Cannot proceed without SMILES.

---

### PHASE 2: Physicochemical Properties & Drug-Likeness

**Goal**: Evaluate whether the compound has drug-like physicochemical properties.

**Steps**:

1. **ADMETAI physicochemical** (primary):
   ```
   ADMETAI_predict_physicochemical_properties(smiles=["<SMILES>"])
   ```
   Returns: MW, logP, TPSA, HBD, HBA, rotatable bonds

2. **SwissADME drug-likeness** (complementary):
   ```
   SwissADME_check_druglikeness(operation="check_druglikeness", smiles="<SMILES>")
   SwissADME_calculate_adme(operation="calculate_adme", smiles="<SMILES>")
   ```
   Returns: Lipinski, Veber, Ghose, Egan, Muegge rule compliance; PAINS alerts; Brenk alerts

3. **ADMETAI solubility**:
   ```
   ADMETAI_predict_solubility_lipophilicity_hydration(smiles=["<SMILES>"])
   ```
   Returns: Aqueous solubility (LogS), lipophilicity, hydration free energy

**Interpret & Score**:

| Property | Ideal Range | Why It Matters |
|----------|-------------|----------------|
| MW | < 500 Da | Larger molecules have poor membrane permeability (Lipinski) |
| LogP | -0.4 to 5.6 | Too hydrophobic = poor solubility; too hydrophilic = poor permeability |
| HBD | <= 5 | Excess donors reduce membrane crossing (Lipinski) |
| HBA | <= 10 | Excess acceptors reduce membrane crossing (Lipinski) |
| TPSA | < 140 A^2 | High PSA correlates with poor oral absorption |
| Rotatable bonds | <= 10 | Molecular flexibility affects bioavailability (Veber) |
| LogS | > -6 | Below -6 = practically insoluble, formulation challenge |
| PAINS alerts | 0 | Pan-assay interference compounds give false positives in screens |

**Verdict**: PASS if Lipinski <= 1 violation and no PAINS alerts; WARN if 2 violations; FAIL if 3+ violations or PAINS+.

**Fallback**: If ADMETAI import fails (missing `tooluniverse[ml]`), rely on SwissADME alone. SwissADME provides all Lipinski descriptors independently.

---

### PHASE 3: ADME Predictions

**Goal**: Predict absorption, distribution, metabolism, and excretion behavior.

**Steps**:

1. **Blood-brain barrier penetration**:
   ```
   ADMETAI_predict_BBB_penetrance(smiles=["<SMILES>"])
   ```
   - BBB+ = compound can cross; BBB- = cannot
   - Critical for CNS drugs (must cross) and peripherally-acting drugs (should NOT cross to avoid CNS side effects)

2. **Oral bioavailability**:
   ```
   ADMETAI_predict_bioavailability(smiles=["<SMILES>"])
   ```
   - F20% = at least 20% oral bioavailability; F30% = at least 30%
   - Low bioavailability means the drug is extensively metabolized or poorly absorbed
   - F < 20% generally requires non-oral routes (IV, inhaled, topical)

3. **CYP450 interactions**:
   ```
   ADMETAI_predict_CYP_interactions(smiles=["<SMILES>"])
   ```
   - Reports substrate/inhibitor status for CYP1A2, 2C9, 2C19, 2D6, 3A4
   - **Why CYP matters**: ~75% of drugs are metabolized by CYP enzymes. Inhibiting CYP3A4 (which metabolizes ~50% of drugs) causes dangerous drug-drug interactions (DDIs). CYP2D6 polymorphisms affect ~25% of drugs -- poor metabolizers accumulate toxic levels
   - Substrate of CYP2D6 = pharmacogenomic risk (poor/ultra-rapid metabolizers)
   - Inhibitor of CYP3A4 = high DDI risk (co-administered drugs accumulate)

4. **Clearance and distribution**:
   ```
   ADMETAI_predict_clearance_distribution(smiles=["<SMILES>"])
   ```
   - VDss (volume of distribution): low (<0.7 L/kg) = confined to plasma; high (>1 L/kg) = distributed to tissues
   - Clearance: high clearance = short half-life, frequent dosing needed
   - Plasma protein binding (PPB): >95% bound = narrow therapeutic window, DDI risk from displacement

5. **SwissADME pharmacokinetics** (cross-validation):
   - GI absorption (high/low), P-gp substrate status, skin permeation (logKp)

**Interpret & Score**:

| Property | Flag Condition | Risk |
|----------|---------------|------|
| BBB+ for non-CNS drug | WARN | Potential CNS side effects |
| BBB- for CNS drug | FAIL | Drug will not reach target |
| F < 20% | WARN | Poor oral bioavailability |
| CYP3A4 inhibitor | WARN | High DDI potential |
| CYP2D6 substrate | WARN | Pharmacogenomic variability |
| PPB > 99% | WARN | Very narrow therapeutic window |
| High clearance + low bioavail | FAIL | Drug unlikely to reach therapeutic levels |

**Fallback**: If ADMETAI unavailable, SwissADME provides GI absorption, BBB permeation (yes/no), P-gp substrate, and CYP inhibition predictions.

---

### PHASE 4: Toxicity Assessment

**Goal**: Evaluate safety liabilities from both predicted and experimental sources.

**Steps**:

1. **ADMETAI toxicity predictions** [T3]:
   ```
   ADMETAI_predict_toxicity(smiles=["<SMILES>"])
   ```
   Key endpoints:
   - **AMES**: Mutagenicity (bacterial reverse mutation test). Positive = potential carcinogen; regulatory agencies require AMES testing for all new drugs
   - **DILI**: Drug-induced liver injury risk. Leading cause of drug withdrawal (e.g., troglitazone). Positive = hepatotoxicity concern requiring liver function monitoring
   - **hERG**: hERG potassium channel inhibition. Causes QT prolongation and fatal cardiac arrhythmia. hERG+ = cardiotoxicity liability; multiple drugs withdrawn for this (e.g., terfenadine, cisapride)
   - **ClinTox**: Clinical trial toxicity / FDA withdrawal risk. Trained on drugs that failed trials or were withdrawn for toxicity
   - **LD50_Zhu**: Predicted lethal dose (mg/kg, rat oral). Lower = more acutely toxic
   - **Skin_Reaction**: Dermal sensitization potential. Important for topical drugs
   - **Carcinogens_Lagunin**: Carcinogenicity prediction

2. **Nuclear receptor activity** [T3]:
   ```
   ADMETAI_predict_nuclear_receptor_activity(smiles=["<SMILES>"])
   ```
   - AR (androgen receptor), ER (estrogen receptor), AhR, PPAR-gamma activity
   - Positive = potential endocrine disruption; critical for chronic-use drugs and environmental chemicals

3. **Stress response pathways** [T3]:
   ```
   ADMETAI_predict_stress_response(smiles=["<SMILES>"])
   ```
   - p53 activation = DNA damage response (genotoxicity signal)
   - MMP disruption = mitochondrial toxicity
   - ATAD5 = DNA repair stress
   - HSE = heat shock / protein misfolding stress

4. **PubChemTox experimental data** [T2] (call all in parallel):
   ```
   PubChemTox_get_toxicity_values(cid=<CID>)
   PubChemTox_get_ghs_classification(cid=<CID>)
   PubChemTox_get_acute_effects(cid=<CID>)
   PubChemTox_get_carcinogen_classification(cid=<CID>)
   PubChemTox_get_target_organs(cid=<CID>)
   PubChemTox_get_toxicity_summary(cid=<CID>)
   ```
   - Real animal study data (LD50, LC50, NOAEL) anchors computational predictions
   - GHS classification provides internationally harmonized hazard categories
   - Carcinogen classification from IARC (Group 1/2A/2B), NTP, EPA

**Interpret & Score**:

| Endpoint | Flag Condition | Severity |
|----------|---------------|----------|
| AMES positive | FAIL | Mutagenic; regulatory barrier |
| DILI positive | WARN | Hepatotox risk; requires monitoring |
| hERG positive | FAIL | Cardiac liability; often program-killing |
| ClinTox positive | WARN | Historical failure signal |
| LD50 < 50 mg/kg | FAIL | Highly toxic (GHS Category 1-2) |
| LD50 50-300 mg/kg | WARN | Toxic (GHS Category 3) |
| LD50 > 2000 mg/kg | PASS | Low acute toxicity |
| NR-ER/AR active | WARN | Endocrine disruption |
| p53 active | WARN | Genotoxicity signal |
| IARC Group 1/2A | FAIL | Known/probable carcinogen |

**Fallback**: If ADMETAI unavailable, PubChemTox provides experimental toxicity data for known compounds. For novel compounds without PubChem entries, flag as "no experimental toxicity data available -- computational predictions only."

---

### PHASE 5: Scorecard Assembly & Clinical Context

**Goal**: Aggregate all findings into a structured ADMET scorecard with pass/warn/fail verdicts.

**Steps**:

1. **ChEMBL clinical status** [T1] (if drug has ChEMBL ID):
   ```
   ChEMBL_get_molecule(chembl_id="<CHEMBL_ID>")
   ```
   - Max phase: 4 = approved, 3 = Phase III, 2 = Phase II, 1 = Phase I, 0 = preclinical
   - Ro5 violations from ChEMBL (independent validation of Lipinski)
   - First approval year, indication class, black box warning flag

2. **Build the ADMET Scorecard**:

```
============================================================
ADMET SCORECARD: [Compound Name] ([SMILES])
============================================================

COMPOUND IDENTITY
  Name:           [name]
  PubChem CID:    [CID]
  SMILES:         [SMILES]
  MW:             [value] Da
  Formula:        [formula]
  ChEMBL Phase:   [0-4 or N/A]

------------------------------------------------------------
CATEGORY            VERDICT    KEY FINDINGS
------------------------------------------------------------
Physicochemical     [P/W/F]    MW=[x], LogP=[x], TPSA=[x], Ro5 viol=[n]
Solubility          [P/W/F]    LogS=[x], class=[soluble/moderate/poor]
Absorption          [P/W/F]    GI=[high/low], F=[x]%, P-gp=[Y/N]
Distribution        [P/W/F]    BBB=[+/-], VDss=[x] L/kg, PPB=[x]%
Metabolism          [P/W/F]    CYP3A4=[sub/inh], CYP2D6=[sub/inh]
Excretion           [P/W/F]    Clearance=[high/med/low], T1/2=[est]
Tox: Mutagenicity   [P/W/F]    AMES=[+/-], p53=[+/-]
Tox: Hepatotoxicity [P/W/F]    DILI=[+/-], target organs=[list]
Tox: Cardiotoxicity [P/W/F]    hERG=[+/-]
Tox: Carcinogenicity[P/W/F]    IARC=[class], Lagunin=[+/-]
Tox: Acute          [P/W/F]    LD50=[x] mg/kg, GHS=[cat]
Endocrine           [P/W/F]    NR-ER=[+/-], NR-AR=[+/-]
Clinical Tox        [P/W/F]    ClinTox=[+/-], skin=[+/-]
------------------------------------------------------------
OVERALL             [P/W/F]    [n] PASS, [n] WARN, [n] FAIL
============================================================

VERDICT DEFINITIONS:
  PASS (P) = Within acceptable range; no flags
  WARN (W) = Borderline or manageable risk; monitor/mitigate
  FAIL (F) = Significant liability; may be program-limiting

EVIDENCE SOURCES:
  [T1] ChEMBL clinical data, FDA labels
  [T2] PubChemTox experimental (LD50, AMES, GHS)
  [T3] ADMETAI predictions, SwissADME calculations
```

3. **Interpretation narrative**: After the scorecard, provide a 3-5 sentence summary:
   - Highlight the most critical findings (any FAILs or WARNs)
   - State whether the compound is suitable for oral administration
   - Note any DDI risks from CYP interactions
   - Flag pharmacogenomic concerns (CYP2D6 substrate)
   - Recommend next steps (e.g., "hERG patch clamp assay recommended to confirm computational prediction")

---

## ADMETAI Dependency Note

ADMETAI tools require the `tooluniverse[ml]` extra:
```bash
pip install tooluniverse[ml]
```

If ADMETAI tools raise `ImportError` or `ModuleNotFoundError`:
1. Log the error: "ADMETAI unavailable -- using fallback tools only"
2. Use SwissADME for physicochemical + drug-likeness + basic PK
3. Use PubChemTox for toxicity (experimental only, no predictions for novel compounds)
4. Mark all prediction-dependent fields as "N/A (ADMETAI not installed)"
5. Still produce the scorecard, marking unavailable rows

---

## Common Query Patterns

### Pattern 1: Named Drug Assessment
```
User: "What are the ADMET properties of metformin?"
Flow: PubChem name->CID(4091)->SMILES -> all 5 phases -> scorecard
```

### Pattern 2: Novel Compound by SMILES
```
User: "Predict toxicity for CC(=O)Nc1ccc(O)cc1"
Flow: Validate SMILES -> PubChem CID lookup -> all 5 phases -> scorecard
Note: PubChemTox may have no data for truly novel compounds
```

### Pattern 3: BBB-Specific Query
```
User: "Can donepezil cross the blood-brain barrier?"
Flow: Resolve SMILES -> Phase 3 BBB only -> brief answer with context
Note: Still report full scorecard if user wants comprehensive profile
```

### Pattern 4: CYP/DDI Risk Query
```
User: "What CYP interactions does ketoconazole have?"
Flow: Resolve SMILES -> Phase 3 CYP only -> report substrate/inhibitor for each isoform
Note: Ketoconazole is a classic CYP3A4 inhibitor -- predictions should match known pharmacology
```

### Pattern 5: Comparative Profiling
```
User: "Compare ADMET of ibuprofen vs naproxen"
Flow: Run full pipeline for each compound -> side-by-side scorecard
Note: Use same SMILES list in ADMETAI calls for efficiency: smiles=["<SMILES1>", "<SMILES2>"]
```

---

## Completeness Checklist (MANDATORY before reporting)

Before delivering the final scorecard, verify:

- [ ] Compound identity resolved (name, CID, SMILES all present or explicitly noted as unavailable)
- [ ] Physicochemical properties reported with Lipinski verdict
- [ ] At least one source for each ADME property (ADMETAI or SwissADME)
- [ ] All 7 ADMETAI toxicity endpoints reported (or marked N/A with reason)
- [ ] PubChemTox experimental data checked (even if "no data found")
- [ ] Nuclear receptor and stress response checked (or marked N/A)
- [ ] Evidence tier tagged for every finding
- [ ] Scorecard table complete with verdicts for all 13 categories
- [ ] Overall verdict stated
- [ ] Interpretation narrative provided with actionable next steps
