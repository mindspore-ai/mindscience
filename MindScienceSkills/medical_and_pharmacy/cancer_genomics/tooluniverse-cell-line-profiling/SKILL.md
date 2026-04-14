---
name: tooluniverse-cell-line-profiling
description: Help researchers select and characterize cancer cell lines for experiments. Given a cancer type, gene of interest, or cell line name, profiles molecular features (mutations, expression, CNV), gene dependencies (CRISPR screens), drug sensitivities (IC50/AUC), and genetic backgrounds using DepMap, Cellosaurus, PharmacoDB, COSMIC, CellMarker, CLUE, and SYNERGxDB. Generates a decision-support report for cell line selection. Use when researchers ask about which cell line to use, cell line characterization, DepMap dependencies, drug sensitivity profiles, or cancer model selection.
license: MIT License
metadata:
    skill-author: ToolUniverse (https://github.com/mims-harvard/TxAgent)
---

# Cancer Cell Line Profiling and Selection

Comprehensive profiling of cancer cell lines for experimental model selection. Transforms a query (cancer type, gene, or cell line name) into an actionable report covering identity verification, molecular features, gene dependencies, drug sensitivities, and druggable targets.

**KEY PRINCIPLES**:
1. **Decision-first** - Answer "which cell line should I use?" not "here is all the data"
2. **Multi-source validation** - Cross-reference DepMap, Cellosaurus, COSMIC, PharmacoDB
3. **Actionable output** - Ranked cell line recommendations with rationale
4. **Practical focus** - Include availability, growth characteristics, common pitfalls
5. **Gene-aware** - When a gene of interest is given, prioritize lines with relevant mutations/dependencies
6. **Source-referenced** - Cite database sources for every claim
7. **English-first queries** - Always use English terms in tool calls, even if the user writes in another language

---

## When to Use

Apply when user asks:
- "Which cell line should I use for NSCLC with EGFR mutations?"
- "Profile A549 for my lung cancer study"
- "What are the gene dependencies of MCF7?"
- "Which breast cancer cell lines are sensitive to paclitaxel?"
- "Does HeLa have TP53 mutations?"
- "Compare A549 vs H1975 for EGFR studies"
- "What drugs work against KRAS-mutant colorectal cancer cell lines?"

---

## Phase 0: Tool Parameter Reference (CRITICAL)

**BEFORE calling ANY tool**, verify parameters against this table.

| Tool | Key Parameters | Notes |
|------|---------------|-------|
| `DepMap_search_cell_lines` | `query` (required) | Search by name, e.g., "A549", "MCF" |
| `DepMap_get_cell_line` | `model_name` OR `model_id` | Name: "A549"; ID: "SIDM00001" |
| `DepMap_get_cell_lines` | `tissue`, `cancer_type`, `page_size` | Filter by tissue (e.g., "Lung") |
| `DepMap_get_gene_dependencies` | `gene_symbol` (required), `model_id` | Gene effect scores; negative = essential |
| `DepMap_search_genes` | `query` (required) | Validate gene symbol in DepMap first |
| `cellosaurus_search_cell_lines` | `q` (required), `size` | Solr syntax: `id:HeLa`, `ox:9606 AND char:cancer` |
| `cellosaurus_get_cell_line_info` | `accession` (required, CVCL_ format) | Full cell line record |
| `cellosaurus_query_converter` | `query` (required) | Natural language to Solr syntax |
| `COSMIC_search_mutations` | `terms` OR `query`, `max_results` | Search "BRAF V600E" or gene name |
| `COSMIC_get_mutations_by_gene` | `gene` OR `gene_name`, `max_results` | All mutations for a gene |
| `PharmacoDB_get_cell_line` | `operation="get_cell_line"`, `cell_name` | Cell line metadata + datasets |
| `PharmacoDB_get_experiments` | `operation="get_experiments"`, `compound_name`, `cell_line_name`, `dataset_name`, `per_page` | Drug response data (IC50, AAC, EC50) |
| `PharmacoDB_get_biomarker_assoc` | `operation="get_biomarker_associations"`, `compound_name`, `tissue_name`, `mdata_type`, `per_page` | Gene-drug sensitivity correlations |
| `PharmacoDB_search` | `operation="search"`, `query` | Find PharmacoDB IDs |
| `CellMarker_search_cancer_markers` | `operation="search_cancer_markers"`, `cancer_type`, `gene_symbol`, `cell_type` | Cancer cell markers |
| `CellMarker_search_by_gene` | `operation="search_by_gene"`, `gene_symbol` (required), `species` | Cell types expressing a gene |
| `HPA_get_comparative_expression_by_gene_and_cellline` | `gene_name` (required), `cell_line` (required) | Supported lines: ishikawa, hela, mcf7, a549, hepg2, jurkat, pc3, rh30, siha, u251 |
| `CLUE_get_cell_lines` | `operation="get_cell_lines"`, `cell_id` | L1000 CMap cell line info (requires CLUE_API_KEY) |
| `SYNERGxDB_search_combos` | `drug_name_1`, `drug_name_2`, `sample` (tissue or cell ID) | Drug combination synergy (ZIP, Bliss, Loewe) |
| `SYNERGxDB_list_cell_lines` | - | All cell lines in SYNERGxDB |
| `DGIdb_get_drug_gene_interactions` | `genes: list[str]` | Druggable gene interactions |
| `OpenTargets_get_associated_drugs_by_target_ensemblID` | `ensemblId`, `size` | Drugs targeting a gene |
| `STRING_get_network` | `protein_ids: list[str]`, `species: int` (9606) | PPI network for gene context |
| `MyGene_query_genes` | `query` (NOT `q`) | Resolve gene symbol to Ensembl ID |
| `cBioPortal_get_mutations` | `study_id`, `gene_list` (STRING, not array) | Cell line mutations from CCLE |

---

## Workflow Overview

```
Input: Cancer type AND/OR Gene of interest AND/OR Cell line name(s)

Phase 1: Cell Line Identification
  - Search and verify cell line identity (Cellosaurus)
  - Get metadata: species, disease, STR profile, cross-references
  - If cancer type given without cell line: find candidate lines (DepMap)

Phase 2: Molecular Profiling
  - Mutation landscape (COSMIC, cBioPortal CCLE)
  - Gene expression (HPA, DepMap)
  - Cancer markers (CellMarker)

Phase 3: Gene Dependencies (CRISPR Screens)
  - Gene essentiality scores from DepMap
  - Identify selectively essential genes
  - Compare across cell lines if multiple candidates

Phase 4: Drug Sensitivity
  - IC50/AAC from PharmacoDB (GDSC, CCLE, CTRPv2, PRISM)
  - Biomarker associations for drug response
  - Drug combination synergy (SYNERGxDB)

Phase 5: Target Druggability & Recommendations
  - Druggable targets (DGIdb, OpenTargets)
  - Final ranked recommendation with rationale
```

---

## Phase 1: Cell Line Identification

**Goal**: Verify cell line identity and find candidates matching the user's needs.

### If user provides a specific cell line name:

1. **Search Cellosaurus** for identity verification:
   ```
   cellosaurus_search_cell_lines(q="id:<CELL_LINE_NAME>", size=5)
   ```
   Extract: accession (CVCL_XXXX), species, disease, synonyms, cross-references.

2. **Get detailed info** with the accession:
   ```
   cellosaurus_get_cell_line_info(accession="CVCL_XXXX")
   ```
   Check: species (must be human for most studies), STR profile, known contamination/misidentification flags.

3. **Get DepMap metadata**:
   ```
   DepMap_get_cell_line(model_name="A549")
   ```
   Extract: tissue, cancer_type, MSI status, ploidy, mutational burden.

4. **Get PharmacoDB metadata**:
   ```
   PharmacoDB_get_cell_line(operation="get_cell_line", cell_name="A549")
   ```
   Extract: tissue, diseases, datasets available, synonyms across studies.

### If user provides cancer type without a specific cell line:

1. **Search DepMap by tissue/cancer type**:
   ```
   DepMap_get_cell_lines(tissue="Lung", page_size=20)
   ```

2. **Narrow by gene of interest** (if provided): after getting candidate list, check which lines carry relevant mutations or dependencies (Phase 2-3).

3. **Cross-reference with CellMarker** for cancer markers:
   ```
   CellMarker_search_cancer_markers(operation="search_cancer_markers", cancer_type="Lung")
   ```

**OUTPUT**: Table of candidate cell lines with: name, tissue, cancer type, key identifiers.

---

## Phase 2: Molecular Profiling

**Goal**: Characterize the mutational and expression landscape of candidate cell lines.

### 2A: Mutation Landscape

1. **COSMIC mutations** for gene of interest:
   ```
   COSMIC_get_mutations_by_gene(operation="get_by_gene", gene="EGFR", max_results=100)
   ```
   Filter results by cell line context if available.

2. **cBioPortal CCLE mutations** (cell line genomics):
   ```
   cBioPortal_get_mutations(study_id="ccle_broad_2019", gene_list="EGFR,KRAS,TP53")
   ```
   NOTE: `gene_list` is a comma-separated STRING, not an array.
   NOTE: `cBioPortal_get_cancer_studies` is BROKEN (literal `{limit}` in URL). Use known study IDs:
   - CCLE: `ccle_broad_2019`

### 2B: Gene Expression (limited to HPA-supported lines)

For HPA-supported cell lines (hela, mcf7, a549, hepg2, jurkat, pc3, rh30, siha, u251, ishikawa):
```
HPA_get_comparative_expression_by_gene_and_cellline(gene_name="EGFR", cell_line="a549")
```

### 2C: Cancer Cell Markers

Check if key genes are known cancer markers:
```
CellMarker_search_by_gene(operation="search_by_gene", gene_symbol="EGFR", species="Human")
```

**OUTPUT**: Mutation table (gene, AA change, type) + expression summary per cell line.

---

## Phase 3: Gene Dependencies (CRISPR Screens)

**Goal**: Determine which genes are essential in candidate cell lines.

**IMPORTANT LIMITATION**: The `DepMap_get_gene_dependencies` tool queries the Sanger Cell Model Passports API, which returns gene metadata (HGNC ID, Ensembl ID, symbol) but NOT per-cell-line CRISPR gene effect scores. Full CRISPR dependency data (Chronos scores) is available at depmap.org but not through this API.

**What you CAN do with current tools**:

1. **Validate gene exists in DepMap**:
   ```
   DepMap_search_genes(query="EGFR")
   ```
   This confirms the gene is tracked and returns identifiers.

2. **Get gene metadata**:
   ```
   DepMap_get_gene_dependencies(gene_symbol="EGFR")
   ```
   Returns: gene symbol, HGNC ID, Ensembl ID. Note: the tool returns metadata only, not per-line scores.

3. **Alternative approaches for essentiality data**:
   - **cBioPortal CCLE**: `cBioPortal_get_mutations(study_id="ccle_broad_2019", gene_list="EGFR")` — check if the gene is mutated (mutated oncogenes are often dependencies)
   - **Literature**: `PubMed_search_articles(query="EGFR dependency CRISPR screen [cell_line]")` — find published screen results
   - **Recommend DepMap portal**: For actual CRISPR scores, direct the user to depmap.org/portal where they can download gene effect data

**Interpreting dependency scores** (when obtained from DepMap portal):
- Score < -0.5: gene is likely essential (cell death upon knockout)
- Score ~ 0: gene is not essential
- Score near -1.0: comparable to common essential genes
- **Selective dependency**: essential in some lineages but not others (indicates therapeutic window)

**OUTPUT**: Gene validation + mutation status per cell line. Note DepMap portal as source for full dependency data.

**Computational procedure: DepMap dependency from downloaded data**

When the API can't deliver CRISPR scores, download the data directly.

**Download instructions**: Go to https://depmap.org/portal/download/all/ (free registration required). Download:
- **CRISPRGeneEffect.csv** (~300MB) — Chronos dependency scores per gene per cell line
- **Model.csv** (~2MB) — Cell line metadata (name, lineage, disease, mutations)
- **OmicsSomaticMutations.csv** (optional, ~100MB) — Mutation calls per cell line

```python
# DepMap CRISPR dependency analysis for cell line selection
# Requires: pandas (included in ToolUniverse dependencies)
import pandas as pd
import os

# --- Configuration ---
depmap_dir = "."  # directory with DepMap files
gene_symbol = "KRAS"  # gene of interest
lineage = "Pancreas"  # cancer lineage

effect_file = os.path.join(depmap_dir, "CRISPRGeneEffect.csv")
model_file = os.path.join(depmap_dir, "Model.csv")

if not os.path.exists(effect_file):
    print(f"Download CRISPRGeneEffect.csv from https://depmap.org/portal/download/all/")
else:
    df = pd.read_csv(effect_file, index_col=0)
    meta = pd.read_csv(model_file)

    # Find gene column (format: "KRAS (3845)")
    gene_col = [c for c in df.columns if c.startswith(f"{gene_symbol} (")]
    if gene_col:
        scores = df[gene_col[0]].dropna()

        # Merge with cell line metadata
        merged = pd.DataFrame({'score': scores}).join(
            meta.set_index('ModelID')[['CellLineName', 'OncotreeLineage',
                                        'OncotreePrimaryDisease', 'OncotreeSubtype']]
        )

        # Filter to lineage of interest
        lineage_lines = merged[merged['OncotreeLineage'] == lineage]

        print(f"=== {gene_symbol} dependency in {lineage} cell lines ===")
        print(f"Total {lineage} lines: {len(lineage_lines)}")
        print(f"\nMost {gene_symbol}-dependent {lineage} lines:")
        top = lineage_lines.sort_values('score').head(15)
        for _, row in top.iterrows():
            dep = "ESSENTIAL" if row['score'] < -0.5 else "moderate" if row['score'] < -0.2 else "not essential"
            print(f"  {row['CellLineName']:20s} score={row['score']:.3f}  ({dep})")

        # For cell line selection: most negative = most dependent on this gene
        # Cross-reference with mutation data to find lines with specific mutations
```

**Interpretation**: Most negative Chronos score = most dependent on the gene. For cell line selection:
- Score < -0.5: The gene is essential in this line (good for studying gene function)
- Score < -1.0: Strongly essential (comparable to common essential genes)
- Compare your target lineage vs pan-cancer to assess selectivity

**If DepMap data is unavailable**: Use `cBioPortal_get_mutations(study_id="ccle_broad_2019", gene_list="KRAS")` for mutation data, and rely on the Quick Reference table in this skill for common cell line recommendations.

---

## Phase 4: Drug Sensitivity

**Goal**: Profile drug response data for candidate cell lines.

### 4A: Drug Response (PharmacoDB)

1. **Get dose-response experiments** for a specific drug + cell line:
   ```
   PharmacoDB_get_experiments(operation="get_experiments", compound_name="Erlotinib", cell_line_name="A549", per_page=20)
   ```
   Key metrics: IC50 (potency), AAC (area above curve, overall sensitivity), EC50, Hill slope.

2. **Get all drug responses for a cell line** (to find most effective drugs):
   ```
   PharmacoDB_get_experiments(operation="get_experiments", cell_line_name="MCF7", per_page=50)
   ```

3. **Biomarker associations** (which genes predict drug sensitivity):
   ```
   PharmacoDB_get_biomarker_assoc(operation="get_biomarker_associations", compound_name="Erlotinib", tissue_name="Lung", mdata_type="mutation", per_page=20)
   ```

### 4B: Drug Combination Synergy (SYNERGxDB)

For drug combination experiments:
```
SYNERGxDB_search_combos(drug_name_1="gemcitabine", drug_name_2="erlotinib", sample="lung")
```
Interpretation: positive ZIP scores indicate synergy; negative scores indicate antagonism.

NOTE: SYNERGxDB covers in vitro cytotoxicity screening only. Clinical regimens (FOLFOX, R-CHOP) and most targeted therapies/biologics are not represented.

### 4C: L1000 Connectivity Map (CLUE)

Check perturbation signatures if CLUE_API_KEY is available:
```
CLUE_get_cell_lines(operation="get_cell_lines", cell_id="MCF7")
```

**OUTPUT**: Drug sensitivity table (drug, IC50, AAC, dataset) + synergy data if available.

---

## Phase 5: Target Druggability and Recommendations

**Goal**: Assess whether key genes/targets are druggable and produce final recommendations.

### 5A: Druggability Assessment

1. **DGIdb interactions**:
   ```
   DGIdb_get_drug_gene_interactions(genes=["EGFR", "KRAS"])
   ```

2. **OpenTargets drugs** (requires Ensembl ID from MyGene):
   ```
   MyGene_query_genes(query="EGFR")
   # Extract ensemblId, then:
   OpenTargets_get_associated_drugs_by_target_ensemblID(ensemblId="ENSG00000146648", size=10)
   ```

3. **Protein interaction context**:
   ```
   STRING_get_network(protein_ids=["EGFR"], species=9606)
   ```

### 5B: Final Recommendation

Synthesize all phases into a ranked recommendation. **Don't just score — explain WHY one line is better than another for this specific use case.**

#### Decision Criteria with Concrete Thresholds

| Criterion | Weight | Score 3 (Best) | Score 2 (Acceptable) | Score 1 (Poor) |
|-----------|--------|----------------|---------------------|----------------|
| **Mutation match** | x3 | Exact mutation (e.g., KRAS G12D) | Same gene, different mutation | No mutation in gene of interest |
| **Co-mutation simplicity** | x2 | Few co-mutations (cleaner background) | Moderate co-mutations | Complex background (3+ driver mutations) |
| **Gene dependency** | x2 | DepMap score < -0.5 (essential) | Score -0.5 to -0.2 (moderately essential) | Score > -0.2 (not essential) |
| **Drug sensitivity data** | x1 | In GDSC + CCLE + PRISM (3+ datasets) | In 1-2 datasets | No drug response data |
| **Practical factors** | x1 | Adherent, well-characterized, widely used | Suspension or less common | Hard to culture, contamination-prone |

**Total score** = sum of (criterion score × weight). Max = 27. Rank cell lines by total score.

#### Use-Case-Specific Guidance

The best cell line depends on what you're doing with it:

| Use Case | Key Requirements | Extra Considerations |
|----------|-----------------|---------------------|
| **CRISPR knockout screen** | Adherent growth, good lentiviral transduction, pre-existing Cas9 clones (check Cellosaurus for "-Cas9" derivatives) | Doubling time matters for library coverage; <72h ideal |
| **Drug sensitivity testing** | In PharmacoDB/GDSC, known IC50 for reference compounds | Check SYNERGxDB for combo data |
| **Xenograft model** | Known tumorigenicity in mice, available PDX data | Check if line forms tumors in nude/NSG mice (Cellosaurus often notes this) |
| **Mechanism of action** | Clean genetic background, gene dependency confirmed | Fewer co-mutations = easier to attribute phenotypes |
| **Biomarker discovery** | Isogenic pairs available, well-characterized omics | Check if isogenic knockouts exist (Cellosaurus) |
| **Drug combination** | In SYNERGxDB with combo data, known single-agent responses | ZIP score available for synergy assessment |

#### Cellosaurus Derivative Lines

**Check for pre-made derivatives** — this can save months of lab work:
- `cellosaurus_search_cell_lines(q="ca:<PARENT_LINE>", size=20)` — finds all derivatives
- Look for: Cas9-expressing clones, drug-resistant derivatives, knockout lines, fluorescent reporter lines
- Example: PANC-1-Cas9-554 through PANC-1-Cas9-559 (CVCL_WL48-WL53) are pre-validated Cas9 clones

#### DepMap API Fallbacks

**If DepMap_get_gene_dependencies fails** (common for some genes):
- The Sanger Cell Model Passports API may not index all genes. Note this limitation.
- Recommend the user check DepMap portal (depmap.org) directly for CRISPR dependency data.
- Use `cBioPortal_get_mutations(study_id="ccle_broad_2019", gene_list="<GENE>")` as an alternative source for cell line mutation data.

**OUTPUT**: Ranked cell line table with total scores, per-criterion breakdown, and a text recommendation explaining the top pick and runner-up with biological reasoning.

---

## Common Use Patterns

### Pattern 1: "Which cell line for [cancer] + [gene]?"

Example: "Which cell line should I use for NSCLC with EGFR mutations?"

1. `DepMap_get_cell_lines(tissue="Lung", page_size=20)` -- get NSCLC lines
2. `DepMap_get_gene_dependencies(gene_symbol="EGFR")` -- check dependency
3. `COSMIC_get_mutations_by_gene(operation="get_by_gene", gene="EGFR")` -- mutation landscape
4. `cBioPortal_get_mutations(study_id="ccle_broad_2019", gene_list="EGFR")` -- CCLE mutations
5. `PharmacoDB_get_experiments(operation="get_experiments", compound_name="Erlotinib", per_page=20)` -- drug response
6. Rank lines by: EGFR mutation status + dependency score + drug sensitivity

Typical answer: H1975 (EGFR L858R + T790M, resistant to 1st-gen TKIs), PC-9 (exon 19 del, sensitive to erlotinib), HCC827 (exon 19 del, EGFR-amplified).

### Pattern 2: "Profile cell line X"

Example: "Profile A549 for my study"

1. `cellosaurus_search_cell_lines(q="id:A549", size=3)` -- identity
2. `DepMap_get_cell_line(model_name="A549")` -- DepMap metadata
3. `PharmacoDB_get_cell_line(operation="get_cell_line", cell_name="A549")` -- PharmacoDB metadata
4. `cBioPortal_get_mutations(study_id="ccle_broad_2019", gene_list="KRAS,TP53,STK11,KEAP1,EGFR")` -- key mutations
5. `HPA_get_comparative_expression_by_gene_and_cellline(gene_name="KRAS", cell_line="a549")` -- expression
6. `PharmacoDB_get_experiments(operation="get_experiments", cell_line_name="A549", per_page=30)` -- drug sensitivities
7. Compile comprehensive profile report

### Pattern 3: "Which lines are sensitive to [drug]?"

Example: "Which breast cancer cell lines are sensitive to paclitaxel?"

1. `DepMap_get_cell_lines(tissue="Breast", page_size=30)` -- breast cancer lines
2. `PharmacoDB_get_experiments(operation="get_experiments", compound_name="Paclitaxel", per_page=50)` -- all paclitaxel data
3. `PharmacoDB_get_biomarker_assoc(operation="get_biomarker_associations", compound_name="Paclitaxel", tissue_name="Breast", per_page=20)` -- sensitivity biomarkers
4. Rank by AAC (higher = more sensitive) or IC50 (lower = more sensitive)

### Pattern 4: "Compare cell line A vs B"

Example: "Compare A549 vs H1975 for EGFR studies"

Run Pattern 2 for both lines in parallel, then build a side-by-side comparison table covering: mutations, dependencies, drug responses, expression.

### Pattern 5: "Drug combinations for [cell line]"

Example: "What drug combinations show synergy in MCF7?"

1. `SYNERGxDB_search_combos(sample="breast", per_page=50)` -- breast cancer combos
2. Filter for MCF7 if results include cell line names
3. `PharmacoDB_get_experiments(operation="get_experiments", cell_line_name="MCF7", per_page=50)` -- single-agent baseline
4. Report synergistic pairs with ZIP scores

---

## Quick Reference: Common Cancer Cell Lines by Type

| Cancer Type | Key Cell Lines | Common Mutations |
|-------------|---------------|-----------------|
| NSCLC | A549 (KRAS G12S), H1975 (EGFR L858R/T790M), PC-9 (EGFR del19), HCC827 (EGFR del19/amp), H460 (KRAS Q61H), H1299 (NRAS Q61K, TP53-null) | KRAS, EGFR, TP53, STK11 |
| Breast | MCF7 (ER+/PR+), MDA-MB-231 (TNBC, KRAS G13D), T-47D (ER+), BT-474 (HER2+), SK-BR-3 (HER2+) | PIK3CA, TP53, BRCA1/2 |
| Colorectal | HCT116 (KRAS G13D, MSI-H), SW480 (KRAS G12V), HT-29 (BRAF V600E), Caco-2 (APC), LoVo (KRAS G13D, MSI-H) | APC, KRAS, TP53, BRAF |
| Melanoma | A375 (BRAF V600E), SK-MEL-28 (BRAF V600E), WM266-4 (BRAF V600D), MeWo (WT BRAF) | BRAF, NRAS, TP53 |
| Pancreatic | PANC-1 (KRAS G12D), MIA PaCa-2 (KRAS G12C), AsPC-1 (KRAS G12D), Capan-1 (BRCA2 mut) | KRAS, TP53, CDKN2A, SMAD4 |
| Prostate | PC-3 (AR-negative), LNCaP (AR+, PTEN-null), DU145 (AR-negative), VCaP (AR amp, TMPRSS2-ERG) | AR, PTEN, TP53, RB1 |
| Ovarian | SKOV3 (HER2+, TP53 mut), OVCAR3 (TP53 mut), A2780 (sensitive), A2780cis (cisplatin-resistant) | TP53, BRCA1/2 |
| Leukemia | K562 (CML, BCR-ABL), Jurkat (T-ALL), HL-60 (AML), THP-1 (AML, monocytic) | BCR-ABL, FLT3, NPM1 |
| Glioblastoma | U251 (TP53 mut), U87MG (PTEN-null), T98G (TP53/PTEN mut), LN229 (TP53 mut, PTEN WT) | TP53, PTEN, EGFR, IDH1 |
| Liver | HepG2 (hepatoblastoma, WT TP53), Hep3B (HBV+, TP53-null), Huh7 (HCC, TP53 Y220C) | TP53, CTNNB1, AXIN1 |

---

## Cross-Referencing Cell Line IDs

Different databases use different identifiers for the same cell line. To join results across tools:

| Database | ID Format | Example | How to Find |
|----------|-----------|---------|-------------|
| **DepMap** | SIDM model_id | SIDM00001 | `DepMap_search_cell_lines(query="A549")` |
| **Cellosaurus** | CVCL accession | CVCL_0023 | `cellosaurus_search_cell_lines(q="id:A549")` |
| **cBioPortal** | Sample ID | A549_LUNG | `cBioPortal_get_mutations(study_id="ccle_broad_2019")` |
| **PharmacoDB** | Cell name string | "A549" | `PharmacoDB_search(operation="search", query="A549")` |
| **SYNERGxDB** | Cell line name | A549 | `SYNERGxDB_list_cell_lines()` |

**Joining strategy**: Use the cell line NAME as the common key (most databases accept human-readable names). When names differ (e.g., "HCT 116" vs "HCT116" vs "HCT-116"), search Cellosaurus for synonyms first.

**Mutation-based filtering**: To find cell lines with a SPECIFIC mutation (e.g., KRAS G12D):
1. `cBioPortal_get_mutations(study_id="ccle_broad_2019", gene_list="KRAS")` — get ALL KRAS mutations in CCLE
2. Filter results by `amino_acid_change` containing "G12D"
3. Extract cell line names from matching records
4. Use these names to query other databases

---

## Error Handling

| Issue | Resolution |
|-------|-----------|
| DepMap returns no results for cell line name | Try alternative names: check Cellosaurus synonyms first |
| cBioPortal CCLE study ID unknown | Use `ccle_broad_2019` as default CCLE study |
| PharmacoDB cell line name mismatch | Use `PharmacoDB_search(operation="search", query="<name>")` to find the canonical name |
| HPA cell line not supported | Only 10 lines supported (hela, mcf7, a549, hepg2, jurkat, pc3, rh30, siha, u251, ishikawa). Skip HPA for other lines |
| CLUE requires API key | Skip CLUE tools if CLUE_API_KEY not set; note in report |
| Gene symbol not found in DepMap | Use `DepMap_search_genes(query="<symbol>")` to check aliases |
| Cellosaurus accession pattern | Must be CVCL_XXXX format; search first if you only have a name |
| SYNERGxDB no results for drug combo | Drug may not be in database; SYNERGxDB covers cytotoxic agents, not most targeted therapies |

---

## Completeness Checklist

Before finalizing the report, verify:

- [ ] Cell line identity verified (Cellosaurus or DepMap)
- [ ] Species confirmed as human (unless otherwise specified)
- [ ] Key mutations documented (COSMIC or cBioPortal)
- [ ] Gene dependency assessed (DepMap CRISPR, if gene of interest provided)
- [ ] Drug sensitivity data included (PharmacoDB, at least one dataset)
- [ ] Druggability of key targets checked (DGIdb or OpenTargets)
- [ ] Practical recommendation provided (not just raw data)
- [ ] All claims cite their source database
- [ ] Known limitations noted (missing data, unsupported lines)
