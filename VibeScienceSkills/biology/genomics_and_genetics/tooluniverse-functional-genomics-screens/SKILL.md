---
name: tooluniverse-functional-genomics-screens
description: Interpret results from CRISPR/shRNA genetic screens using DepMap essentiality data, constraint scores, pathway enrichment, protein networks, druggability assessment, and clinical evidence. Use for screen hit validation, gene essentiality analysis, DepMap exploration, functional genomics interpretation, and screen-to-target prioritization.
license: MIT License
metadata:
    skill-author: ToolUniverse (https://github.com/mims-harvard/TxAgent)
---

# Functional Genomics Screen Interpretation

Pipeline for validating and prioritizing hits from genetic screens (CRISPR knockout, CRISPRi, shRNA) by integrating essentiality data (DepMap), evolutionary constraint (gnomAD), pathway context (Reactome, STRING), druggability (DGIdb), and clinical evidence (CIViC, COSMIC). Transforms a list of screen hits into a ranked, actionable target list.

**Guiding principles**:
1. **Hits are hypotheses** -- screen results contain false positives and context-dependent effects; validation through orthogonal evidence is the goal
2. **Selectivity matters** -- a gene essential in all cell lines is less interesting than one essential only in the disease context
3. **Pathway over gene** -- enriched pathways are more robust than individual gene hits; always look for convergent biology
4. **Druggability is practical** -- prioritize hits that can be chemically or biologically modulated
5. **Clinical anchoring** -- connect screen hits to known disease biology via CIViC, COSMIC, and literature
6. **English-first queries** -- use English gene names in tool calls

---

## When to Use

Typical triggers:
- "Here are my CRISPR screen hits -- which ones are real?"
- "Validate these essential genes from my screen"
- "What pathways are enriched in my screen hits?"
- "Which of these hits are druggable?"
- "Check DepMap essentiality for [gene list]"
- "Prioritize screen hits for [cancer type]"
- "Are these genes selectively essential in [context]?"

**Not this skill**: For differential gene expression analysis, use `tooluniverse-rnaseq-deseq2`. For pathway enrichment of expression data, use `tooluniverse-gene-enrichment`. For single-cell analysis, use `tooluniverse-single-cell`.

---

## Core Databases

| Database | Scope | Best For |
|----------|-------|----------|
| **DepMap** | Cancer cell line essentiality (1000+ lines), drug response | Context-specific essentiality, pan-cancer vs selective dependency |
| **gnomAD** | Population genetic constraint scores | Evolutionary importance; loss-of-function intolerance |
| **STRING** | Protein-protein interaction network | Network context, functional associations |
| **Reactome** | Curated biological pathways | Pathway enrichment of hit lists |
| **DGIdb** | Drug-gene interactions and druggability | Target tractability, existing drugs/compounds |
| **CIViC** | Clinical evidence for variants/genes | Clinical relevance, therapeutic implications |
| **COSMIC** | Cancer mutation catalog | Somatic mutation frequency, oncogene/TSG classification |
| **UniProt** | Protein function annotation | Gene function, domains, subcellular localization |

---

## Workflow Overview

```
Phase 0: Input Processing
  Parse hit list, identify gene names, screen type, context
    |
Phase 1: Hit Validation (Essentiality + Constraint)
  DepMap dependency scores, gnomAD constraint, gene function
    |
Phase 2: Pathway & Network Context
  Reactome enrichment, STRING network, functional clusters
    |
Phase 3: Druggability Assessment
  DGIdb interactions, druggable gene categories
    |
Phase 4: Clinical Evidence
  CIViC variants, COSMIC mutations, disease relevance
    |
Phase 5: Literature Context
  PubMed studies on key hits
    |
Phase 6: Prioritized Report
  Ranked target list with multi-dimensional scoring
```

---

## Phase Details

### Phase 0: Input Processing

Parse the user's input to identify:
- **Gene list**: names or IDs of screen hits
- **Screen type**: CRISPR-KO, CRISPRi, CRISPRa, shRNA (affects interpretation)
- **Cell line or model**: what was screened
- **Control comparison**: positive vs negative selection, what phenotype was scored
- **Disease context**: cancer type, tissue, or condition

If the user provides a ranked list (e.g., by MAGeCK score or BAGEL BF), preserve the ranking. If unranked, all hits are treated equally in Phase 1.

### Phase 1: Hit Validation (Essentiality + Constraint)

**Objective**: Assess whether screen hits are genuine essential genes in the relevant context.

**Tools**:
- `DepMap_get_gene_dependencies` -- get gene dependency metadata
  - Input: `gene_symbol` (gene symbol)
  - Output: gene metadata (HGNC ID, Ensembl ID). **NOTE: This tool returns gene catalog metadata, NOT per-cell-line CRISPR scores.** Full Chronos dependency data must be obtained from depmap.org directly. As a workaround, use `search_clinical_trials` and `PubMed_search_articles` to find published CRISPR screen data for specific genes.
- `DepMap_search_cell_lines` -- find cell lines matching criteria
  - Input: `query` (cell line name, lineage, or disease)
  - Output: cell line metadata, lineage, disease
- `gnomad_get_gene_constraints` -- get loss-of-function constraint
  - Input: `gene_symbol`
  - Output: pLI score, LOEUF, missense constraint (may return "Service overloaded" -- treat as transient)
- `UniProt_get_function_by_accession` -- get protein function summary
  - Input: `accession` (UniProt ID)
  - Output: list of function description strings

**Workflow**:
1. For each hit gene, query DepMap for dependency scores
2. Compare dependency in the screen's cell line/lineage vs pan-cancer
3. Classify each hit:
   - **Pan-essential**: essential in >90% of lines (common essential, less interesting for therapy)
   - **Selectively essential**: essential in specific lineages or contexts (high-value therapeutic targets)
   - **Context-specific**: essential only in the screen's model (needs further validation)
4. Get gnomAD constraint: pLI > 0.9 or LOEUF < 0.35 indicates strong evolutionary constraint
5. Get UniProt function for biological context

**Interpreting DepMap scores**:
- Chronos score < -0.5: likely essential
- Chronos score < -1.0: strongly essential
- Compare to median across all lines to assess selectivity

**DepMap API limitation and workaround**: The `DepMap_get_gene_dependencies` tool (Sanger Cell Model Passports API) returns gene metadata only — NOT per-cell-line CRISPR dependency scores. For actual Chronos scores, download data files from the DepMap portal and process locally.

**Step 1: Download DepMap data files**

Go to https://depmap.org/portal/download/all/ and download these files (registration required, free):
- **CRISPRGeneEffect.csv** (~300MB) — Chronos gene effect scores. Rows = cell lines (ACH-XXXXXX), columns = genes ("GENE (ENTREZ_ID)"). Negative = essential.
- **Model.csv** (~2MB) — Cell line metadata: name, lineage, cancer type, mutations.
- **OmicsExpressionProteinCodingGenesTPMLogp1.csv** (optional, ~500MB) — RNA-seq expression.

Alternatively, download via the DepMap Python client:
```bash
pip install depmap-client  # if available
```

Or use direct download links from the DepMap release page (URLs change each quarter — check the portal for the current release, e.g., "DepMap Public 24Q4").

**Step 2: Analyze dependency data**

```python
# Computational procedure: DepMap CRISPR dependency analysis
# Requires: pandas, scipy (both in ToolUniverse dependencies)
# Input: CRISPRGeneEffect.csv and Model.csv from DepMap portal
import pandas as pd
import os

# --- Configuration ---
depmap_dir = "."  # directory containing downloaded DepMap files
gene_symbol = "PTPN11"  # SHP2 - change to your gene of interest
lineage = "Lung"  # OncotreeLineage to test selectivity against

# --- Load data ---
effect_file = os.path.join(depmap_dir, "CRISPRGeneEffect.csv")
model_file = os.path.join(depmap_dir, "Model.csv")

if not os.path.exists(effect_file):
    print(f"ERROR: {effect_file} not found.")
    print("Download from: https://depmap.org/portal/download/all/")
    print("Look for: CRISPR (DepMap Public) → CRISPRGeneEffect.csv")
else:
    df = pd.read_csv(effect_file, index_col=0)
    meta = pd.read_csv(model_file)

    # --- Find gene column ---
    # Columns are "GENE (ENTREZ_ID)" format, e.g., "PTPN11 (5781)"
    gene_col = [c for c in df.columns if c.startswith(f"{gene_symbol} (")]
    if not gene_col:
        # Try case-insensitive
        gene_col = [c for c in df.columns if c.upper().startswith(f"{gene_symbol.upper()} (")]

    if not gene_col:
        print(f"Gene {gene_symbol} not found in DepMap columns")
        print(f"Available genes starting with {gene_symbol[0]}: "
              f"{[c for c in df.columns if c.startswith(gene_symbol[0])][:10]}")
    else:
        scores = df[gene_col[0]].dropna()
        print(f"\n=== {gene_symbol} dependency across {len(scores)} cell lines ===")
        print(f"Median (all): {scores.median():.3f}")
        print(f"Mean (all):   {scores.mean():.3f}")
        print(f"% essential (< -0.5): {(scores < -0.5).mean()*100:.1f}%")
        print(f"% strongly essential (< -1.0): {(scores < -1.0).mean()*100:.1f}%")

        # --- Selective essentiality ---
        # Merge with metadata to split by lineage
        meta_idx = meta.set_index('ModelID')
        lineage_ids = meta[meta['OncotreeLineage'] == lineage]['ModelID']
        in_lineage = scores[scores.index.isin(lineage_ids)]
        out_lineage = scores[~scores.index.isin(lineage_ids)]

        print(f"\n=== Selectivity: {lineage} vs rest ===")
        print(f"{lineage} lines (n={len(in_lineage)}): median={in_lineage.median():.3f}")
        print(f"Other lines (n={len(out_lineage)}):  median={out_lineage.median():.3f}")

        from scipy.stats import mannwhitneyu
        if len(in_lineage) >= 3 and len(out_lineage) >= 3:
            stat, pval = mannwhitneyu(in_lineage, out_lineage, alternative='less')
            print(f"Mann-Whitney U (more essential in {lineage}?): p={pval:.2e}")
            if pval < 0.05 and in_lineage.median() < -0.3:
                print(f"→ SELECTIVELY ESSENTIAL in {lineage} (good therapeutic target)")
            elif scores.median() < -0.5:
                print(f"→ PAN-ESSENTIAL (essential everywhere — high toxicity risk)")
            else:
                print(f"→ NOT ESSENTIAL in most lines")

        # --- Top 10 most dependent cell lines ---
        print(f"\n=== Top 10 most {gene_symbol}-dependent cell lines ===")
        top10 = scores.nsmallest(10)
        for model_id, score in top10.items():
            info = meta_idx.loc[model_id] if model_id in meta_idx.index else {}
            name = info.get('CellLineName', '?') if isinstance(info, pd.Series) else '?'
            lin = info.get('OncotreeLineage', '?') if isinstance(info, pd.Series) else '?'
            print(f"  {name:20s} ({lin:15s}): {score:.3f}")
```

**Step 3: Interpret results**

| Metric | Interpretation | Action |
|--------|---------------|--------|
| Median < -0.5 in ALL lines | Pan-essential (like ribosomal genes) | Poor drug target — will kill healthy cells too |
| Median < -0.5 in lineage, > -0.2 in others | **Selectively essential** | Best drug target — therapeutic window exists |
| Median > -0.2 everywhere | Not essential | Screen hit may be false positive |
| Mann-Whitney p < 0.05 | Selectivity is statistically significant | Supports lineage-specific dependency |
| Top-10 list dominated by one lineage | Strong lineage specificity | Validates the screen's cell line context |

**When DepMap files are not available**: Fall back to gnomAD constraint (pLI, LOEUF) + literature search for published CRISPR screen results. Use `PubMed_search_articles(query="[gene] CRISPR screen essential [cancer_type]")` to find published dependency data.

### Phase 2: Pathway & Network Context

**Objective**: Identify convergent biological themes in the hit list.

**Tools**:
- `ReactomeAnalysis_pathway_enrichment` -- pathway enrichment analysis
  - Input: `identifiers` (space-separated gene list as STRING, not array)
  - Output: enriched pathways with p-values, gene counts
- `STRING_get_network` -- get protein interaction network
  - Input: `identifiers` (carriage-return-separated string: `"GENE1\rGENE2\rGENE3"`, NOT an array), `species` (9606 for human)
  - Output: interaction edges with confidence scores
- `STRING_functional_enrichment` -- GO/KEGG enrichment via STRING
  - Input: `identifiers` (same format as above), `species` (9606)
  - Output: enriched terms with FDR-corrected p-values

**Workflow**:
1. Run Reactome pathway enrichment on the full hit list
2. Build STRING network for the top hits (limit to 20-50 genes for interpretable networks)
3. Run STRING functional enrichment for GO terms and KEGG pathways
4. Identify pathway clusters -- groups of hits converging on the same biology
5. Highlight hits that are network hubs (high connectivity)

**Important**: For `ReactomeAnalysis_pathway_enrichment`, identifiers are space-separated (e.g., `"TP53 BRCA1 EGFR"`), not a list.

### Phase 3: Druggability Assessment

**Objective**: Determine which hits can be therapeutically targeted.

**Tools**:
- `DGIdb_get_drug_gene_interactions` -- find drugs targeting hit genes
  - Input: `genes` (list of gene names, e.g., `["EGFR", "BRAF"]`)
  - Output: drug-gene interactions with drug names, interaction types, sources
- `DGIdb_get_gene_druggability` -- assess druggable gene categories
  - Input: `genes` (list of gene names)
  - Output: `{data: {genes: {nodes: [{name, geneCategories}]}}}`
- `PharmacoDB_get_experiments` -- get drug sensitivity data
  - Input: `drug_name` or `gene_name`
  - Output: drug response (IC50, AUC) across cell lines

**Workflow**:
1. Query DGIdb for all hit genes to find existing drug-gene interactions
2. Get druggability categories (kinase, GPCR, ion channel, etc.)
3. For druggable hits, check DepMap drug response data
4. Classify each hit:
   - **Approved drug exists**: direct translational potential
   - **Druggable category, no approved drug**: compounds may exist in development
   - **Not in druggable category**: may need novel modalities (PROTAC, antisense, etc.)
5. **DGIdb may lag clinical reality** — for high-priority hits where DGIdb shows few/no drugs, also search `search_clinical_trials(query_term="<gene_symbol> inhibitor")` and `PubMed_search_articles(query="<gene_symbol> inhibitor clinical trial")`. Novel allosteric inhibitors (e.g., SHP2, SOS1, KRAS G12D) may be in Phase I/II trials but not yet in DGIdb.

### Phase 4: Clinical Evidence

**Objective**: Connect screen hits to known clinical and cancer biology.

**Tools**:
- `civic_search_evidence_items` -- search CIViC for clinical evidence
  - Input: `molecular_profile` (gene name or variant, NOT `query`), optional `therapy`, `disease`
  - Output: evidence items with clinical significance, disease, drugs
- `COSMIC_get_mutations_by_gene` -- get somatic mutation data
  - Input: `gene_name`
  - Output: mutation types, frequencies, cancer types
- `PubMed_search_articles` -- find relevant publications
  - Input: `query`, optional `limit`
  - Output: list of article dicts

**Workflow**:
1. For top prioritized hits, search CIViC for clinical evidence
2. Query COSMIC for somatic mutation frequency (high frequency = likely driver)
3. Classify genes as: known oncogene, tumor suppressor, or novel
4. Note any existing clinical implications (predictive, prognostic, diagnostic)

### Phase 5: Literature Context

**Objective**: Find published studies on key hits, especially in the disease context.

**Tools**:
- `PubMed_search_articles` -- biomedical literature search
  - Input: `query`, optional `limit`
  - Output: list of article dicts

**Workflow**:
1. For the top 5-10 prioritized hits, search PubMed for "[gene] [disease] essentiality" or "[gene] CRISPR screen"
2. Look for validation studies, mechanism papers, and therapeutic targeting efforts
3. Note any conflicting evidence or context-dependent findings

### Phase 6: Prioritized Report

Generate a ranked target list integrating all dimensions:

**Quantitative hit prioritization score (0-30)**:

| Criterion | Score 3 | Score 2 | Score 1 | Score 0 |
|-----------|---------|---------|---------|---------|
| **Selective essentiality** | DepMap score < -0.5 in disease context AND > -0.2 in others | < -0.5 in disease, < -0.5 in some others | < -0.5 pan-essential | > -0.2 (not essential) |
| **Pathway convergence** | 3+ hits in same pathway | 2 hits in same pathway | Gene in enriched pathway | Isolated hit, no pathway context |
| **Druggability** | Approved drug exists | Druggable category (kinase, GPCR) | Chemical probes available | Not druggable |
| **Clinical evidence** | CIViC therapeutic evidence | COSMIC driver gene (oncogene/TSG) | CIViC any evidence level | No clinical data |
| **Constraint** | pLI > 0.9 (LOF intolerant) | pLI 0.5-0.9 | pLI < 0.5 but gene is annotated | No constraint data |
| **Literature** | Multiple validation studies in context | 1 validation study | Published in different context | No publications |

**Total**: Sum all rows. Max = 18. **Tier 1** (15-18): high-confidence target. **Tier 2** (10-14): promising, needs validation. **Tier 3** (5-9): speculative. **Tier 4** (<5): likely false positive or biologically uninteresting.

**Key interpretation**: Pan-essential genes (score 0 in selectivity) are usually NOT good drug targets despite being "essential" — they are essential in healthy cells too, predicting toxicity. The most valuable hits have **selective** essentiality in the disease context.

**Report structure**:
1. **Screen Summary** -- type, model, phenotype, total hits
2. **Top Targets** (ranked table) -- gene, DepMap score, selectivity, constraint, druggability, clinical evidence
3. **Enriched Pathways** -- convergent biology themes with gene members
4. **Network Analysis** -- interaction hubs, pathway clusters
5. **Druggable Targets** -- compounds, interaction types, development status
6. **Clinical Connections** -- CIViC evidence, COSMIC mutations
7. **Validation Recommendations** -- suggested orthogonal experiments
8. **Data Gaps** -- genes with limited annotation, missing data

---

## Common Analysis Patterns

| Pattern | Description | Key Phases |
|---------|-------------|------------|
| **Full Screen Interpretation** | Comprehensive hit validation and prioritization | All (0-6) |
| **Essentiality Check** | Quick DepMap lookup for a gene list | 0, 1 |
| **Pathway Discovery** | What biology do my hits converge on? | 0, 2 |
| **Druggable Target ID** | Which hits can I target with drugs? | 0, 1, 3 |
| **Clinical Translation** | Connect screen to patient relevance | 0, 1, 4, 5 |

---

## Edge Cases & Fallbacks

- **gnomAD "Service overloaded"**: Transient error. Retry once, then proceed without constraint data and note the gap
- **Gene not in DepMap**: Possible if the gene is not well-annotated. Fall back to gnomAD constraint and UniProt function
- **Large hit lists (>500 genes)**: Focus pathway enrichment on the full list, but do per-gene analysis only on top 50 by screen score
- **Non-cancer screens**: DepMap is cancer-focused. For non-cancer contexts, DepMap selectivity is less informative; weight constraint and pathway evidence more heavily
- **shRNA vs CRISPR**: shRNA screens have more off-target effects. Higher validation bar for shRNA-only hits

---

## Limitations

- **DepMap is cancer-centric**: Essentiality data covers ~1000 cancer cell lines; limited utility for non-cancer biology
- **Screen artifacts**: Seed effects (shRNA), copy number bias (CRISPR), and fitness confounders are not assessed by this skill
- **No raw screen analysis**: This skill interprets hit lists, not raw screen data (read counts, guide-level data). Use MAGeCK/BAGEL/CRISPRcleanR for upstream analysis
- **COSMIC access**: Some COSMIC data requires authentication; results may be limited
- **Interaction directionality**: STRING interactions are associations, not causal relationships
