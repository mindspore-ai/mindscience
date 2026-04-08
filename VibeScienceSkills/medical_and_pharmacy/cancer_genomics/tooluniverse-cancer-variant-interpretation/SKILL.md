---
name: tooluniverse-cancer-variant-interpretation
description: Provide comprehensive clinical interpretation of somatic mutations in cancer. Given a gene symbol + variant (e.g., EGFR L858R, BRAF V600E) and optional cancer type, performs multi-database analysis covering clinical evidence (CIViC), mutation prevalence (cBioPortal), therapeutic associations (OpenTargets, ChEMBL, FDA), resistance mechanisms, clinical trials, prognostic impact, and pathway context. Generates an evidence-graded markdown report with actionable recommendations for precision oncology. Use when oncologists, molecular tumor boards, or researchers ask about treatment options for specific cancer mutations, resistance mechanisms, or clinical trial matching.
license: MIT License
metadata:
    skill-author: ToolUniverse (https://github.com/mims-harvard/TxAgent)
---

# Cancer Variant Interpretation for Precision Oncology

Comprehensive clinical interpretation of somatic mutations in cancer. Transforms a gene + variant input into an actionable precision oncology report covering clinical evidence, therapeutic options, resistance mechanisms, clinical trials, and prognostic implications.

**KEY PRINCIPLES**:
1. **Report-first approach** - Create report file FIRST, then populate progressively
2. **Evidence-graded** - Every recommendation has an evidence tier (T1-T4)
3. **Actionable output** - Prioritized treatment options, not data dumps
4. **Clinical focus** - Answer "what should we treat with?" not "what databases exist?"
5. **Resistance-aware** - Always check for known resistance mechanisms
6. **Cancer-type specific** - Tailor all recommendations to the patient's cancer type when provided
7. **Source-referenced** - Every statement must cite the tool/database source
8. **English-first queries** - Always use English terms in tool calls (gene names, drug names, cancer types), even if the user writes in another language. Respond in the user's language

---

## When to Use

Apply when user asks:
- "What treatments exist for EGFR L858R in lung cancer?"
- "Patient has BRAF V600E melanoma - what are the options?"
- "Is KRAS G12C targetable?"
- "Patient progressed on osimertinib - what's next?"
- "What clinical trials are available for PIK3CA E545K?"
- "Interpret this somatic mutation: TP53 R273H"

---

## Input Parsing

**Required**: Gene symbol + variant notation
**Optional**: Cancer type (improves specificity)

### Accepted Input Formats

| Format | Example | How to Parse |
|--------|---------|-------------|
| Gene + amino acid change | EGFR L858R | gene=EGFR, variant=L858R |
| Gene + HGVS protein | BRAF p.V600E | gene=BRAF, variant=V600E |
| Gene + exon notation | EGFR exon 19 deletion | gene=EGFR, variant=exon 19 deletion |
| Gene + fusion | EML4-ALK fusion | gene=ALK, variant=EML4-ALK |
| Gene + amplification | HER2 amplification | gene=ERBB2, variant=amplification |

### Gene Symbol Normalization

Common aliases: HER2 -> ERBB2, PD-L1 -> CD274, VEGF -> VEGFA

---

## Phase 0: Tool Parameter Verification (CRITICAL)

**BEFORE calling ANY tool for the first time**, verify its parameters.

| Tool | WRONG Parameter | CORRECT Parameter |
|------|-----------------|-------------------|
| `OpenTargets_get_associated_drugs_by_target_ensemblID` | `ensemblID` | `ensemblId` (camelCase) |
| `OpenTargets_get_drug_chembId_by_generic_name` | `genericName` | `drugName` |
| `OpenTargets_target_disease_evidence` | `ensemblID` | `ensemblId` + `efoId` |
| `MyGene_query_genes` | `q` | `query` |
| `search_clinical_trials` | `disease`, `biomarker` | `condition`, `query_term` (required) |
| `civic_get_variants_by_gene` | `gene_symbol` | `gene_id` (CIViC numeric ID) |
| `drugbank_*` | any 3 params | ALL 4 required: `query`, `case_sensitive`, `exact_match`, `limit` |
| `ChEMBL_get_drug_mechanisms` | `chembl_id` | `drug_chembl_id__exact` |
| `ensembl_lookup_gene` | no species | `species='homo_sapiens'` is REQUIRED |

---

## Workflow Overview

```
Input: Gene symbol + Variant notation + Optional cancer type

Phase 1: Gene Disambiguation & ID Resolution
  - Resolve gene to Ensembl ID, UniProt accession, Entrez ID
  - Get gene function, pathways, protein domains
  - Identify cancer type EFO ID (if cancer type provided)

Phase 2: Clinical Variant Evidence (CIViC)
  - Find gene in CIViC (via Entrez ID matching)
  - Get all variants for the gene, match specific variant
  - Retrieve evidence items (predictive, prognostic, diagnostic)

Phase 3: Mutation Prevalence (cBioPortal)
  - Frequency across cancer studies
  - Co-occurring mutations, cancer type distribution

Phase 4: Therapeutic Associations (OpenTargets + ChEMBL + FDA + DrugBank)
  - FDA-approved targeted therapies
  - Clinical trial drugs (phase 2-3), drug mechanisms
  - Combination therapies

Phase 5: Resistance Mechanisms
  - Known resistance variants (CIViC, literature)
  - Bypass pathway analysis (Reactome)

Phase 6: Clinical Trials
  - Active trials recruiting for this mutation
  - Trial phase, status, eligibility

Phase 7: Prognostic Impact & Pathway Context
  - Survival associations (literature)
  - Pathway context (Reactome), Expression data (GTEx)

Phase 8: Report Synthesis
  - Executive summary, clinical actionability score
  - Treatment recommendations (prioritized), completeness checklist
```

For detailed code snippets and API call patterns for each phase, see `ANALYSIS_DETAILS.md`.

---

## Evidence Grading Summary

| Tier | Criteria | Examples |
|------|----------|---------|
| **T1** | FDA-approved therapy, Level A CIViC evidence, phase 3 trial | Osimertinib for EGFR T790M |
| **T2** | Phase 2/3 clinical data, Level B CIViC evidence | Combination trial data |
| **T3** | Preclinical data, Level D CIViC, case reports | Novel mechanisms, in vitro |
| **T4** | Computational prediction, pathway inference | Docking, pathway analysis |

## Clinical Actionability Scoring

| Score | Criteria |
|-------|----------|
| **HIGH** | FDA-approved targeted therapy for this exact mutation + cancer type |
| **MODERATE** | Approved therapy for different cancer type with same mutation, OR phase 2-3 trial data |
| **LOW** | Only preclinical evidence or pathway-based rationale |
| **UNKNOWN** | Insufficient data to assess actionability |

For full scoring tables and treatment prioritization, see `SCORING_TABLES.md`.

---

## Tool Reference (Verified Parameters)

### Gene Resolution

| Tool | Key Parameters | Response Key Fields |
|------|---------------|-------------------|
| `MyGene_query_genes` | `query`, `species` | `hits[].ensembl.gene`, `.entrezgene`, `.symbol` |
| `UniProt_search` | `query`, `organism`, `limit` | `results[].accession` |
| `OpenTargets_get_target_id_description_by_name` | `targetName` | `data.search.hits[].id` |
| `ensembl_lookup_gene` | `gene_id`, `species` (REQUIRED) | `data.id`, `.version` |

### Clinical Evidence

| Tool | Key Parameters | Response Key Fields |
|------|---------------|-------------------|
| `civic_search_genes` | `query`, `limit` | `data.genes.nodes[].id`, `.entrezId` |
| `civic_get_variants_by_gene` | `gene_id` (CIViC numeric) | `data.gene.variants.nodes[]` |
| `civic_get_variant` | `variant_id` | `data.variant` |

### Drug Information

| Tool | Key Parameters | Response Key Fields |
|------|---------------|-------------------|
| `OpenTargets_get_associated_drugs_by_target_ensemblID` | `ensemblId`, `size` | `data.target.knownDrugs.rows[]` |
| `FDA_get_indications_by_drug_name` | `drug_name`, `limit` | `results[].indications_and_usage` |
| `drugbank_get_drug_basic_info_by_drug_name_or_id` | `query`, `case_sensitive`, `exact_match`, `limit` (ALL required) | `results[]` |

### Mutation Prevalence

| Tool | Key Parameters | Response Key Fields |
|------|---------------|-------------------|
| `cBioPortal_get_mutations` | `study_id`, `gene_list` | `data[].proteinChange` |
| `cBioPortal_get_cancer_studies` | `limit` | `[].studyId`, `.cancerTypeId` |

### Clinical Trials & Literature

| Tool | Key Parameters | Response Key Fields |
|------|---------------|-------------------|
| `search_clinical_trials` | `query_term` (required), `condition` | `studies[]` |
| `PubMed_search_articles` | `query`, `limit`, `include_abstract` | Returns **list** of dicts (NOT wrapped) |
| `Reactome_map_uniprot_to_pathways` | `id` (UniProt accession) | Pathway mappings |
| `GTEx_get_median_gene_expression` | `gencode_id`, `operation="median"` | Expression by tissue |

For full tool parameter reference, see `TOOLS_REFERENCE.md`.

---

## Fallback Chains

| Primary Tool | Fallback | Use When |
|-------------|----------|----------|
| CIViC variant lookup | PubMed literature search | Gene not found in CIViC |
| OpenTargets drugs | ChEMBL drug search | No OpenTargets drug hits |
| FDA indications | DrugBank drug info | Drug not in FDA database |
| cBioPortal TCGA study | cBioPortal pan-cancer | Specific cancer study not available |
| GTEx expression | Ensembl gene lookup | GTEx returns empty |
| Reactome pathways | UniProt function | Pathway mapping fails |

---

## Quantified Minimums

| Section | Requirement |
|---------|-------------|
| Gene IDs | At least Ensembl + UniProt resolved |
| Clinical evidence | CIViC queried + PubMed literature search |
| Mutation prevalence | At least 1 cBioPortal study |
| Therapeutic options | All approved drugs listed + FDA label for top drugs |
| Resistance | Literature search + known patterns documented |
| Clinical trials | At least 1 search query executed |
| Prognostic impact | PubMed literature search performed |
| Pathway context | Reactome pathway mapping attempted |

---

## See Also

- `ANALYSIS_DETAILS.md` - Detailed code snippets and API call patterns for each phase
- `REPORT_TEMPLATE.md` - Full report template with completeness checklist
- `SCORING_TABLES.md` - Evidence grading, treatment prioritization, use cases
- `TOOLS_REFERENCE.md` - Detailed tool parameter reference
- `QUICK_START.md` - Example usage and quick reference
- `EXAMPLES.md` - Complete example reports
