---
name: tooluniverse-metagenomics-analysis
description: Analyze microbiome and metagenomics data using MGnify, GTDB, ENA, and literature tools. Search studies by biome/keyword, retrieve taxonomic profiles and functional annotations, classify genomes with GTDB taxonomy, and find related publications. Use for human gut microbiome, soil/ocean metagenomics, and environmental microbiology research.
license: MIT License
metadata:
    skill-author: ToolUniverse (https://github.com/mims-harvard/TxAgent)
---

# Metagenomics & Microbiome Analysis

Integrated pipeline for exploring microbiome studies, classifying taxa, assessing genome quality, linking microbial composition to clinical phenotypes, and interpreting findings through pathway analysis and literature context.

**Guiding principles**:
1. **Study context first** -- understand the biome, sequencing method, and sample metadata before diving into taxa
2. **Taxonomic consistency** -- use GTDB taxonomy as the reference standard; reconcile NCBI taxonomy where needed
3. **Genome quality matters** -- CheckM completeness/contamination thresholds determine which MAGs are trustworthy
4. **Interpretation over enumeration** -- don't just list taxa; explain what they mean for the biological question
5. **Clinical grounding** -- connect microbial signatures to human phenotypes via GMrepo and literature
6. **Progressive reporting** -- create the report file early and update it as phases complete
7. **English-first queries** -- use English terms in tool calls; respond in the user's language

---

## When to Use

Typical triggers:
- "What microbiome studies exist for [biome/disease]?"
- "Classify these taxa using GTDB"
- "Find metagenome-assembled genomes for [species]"
- "What gut bacteria are associated with [phenotype]?"
- "Search for 16S rRNA studies on [topic]"
- "Assess genome quality for [accession]"
- "Compare microbiome composition in [condition A] vs [condition B]"

**Not this skill**: For host genomics or variant calling, use `tooluniverse-variant-analysis`. For pathway enrichment of host genes, use `tooluniverse-gene-enrichment`.

---

## Core Databases

| Database | Scope | Best For |
|----------|-------|----------|
| **MGnify** | EBI metagenomics portal; studies, analyses, genomes, biomes | Discovering processed metagenomics studies and their taxonomic/functional results |
| **GTDB** | Genome Taxonomy Database; standardized bacterial/archaeal taxonomy | Consistent taxonomic classification, species-level resolution |
| **GMrepo** | Gut Microbiome Repository; curated phenotype-microbe associations | Linking gut species to human health conditions |
| **ENA** | European Nucleotide Archive; raw sequencing data | Finding raw 16S/shotgun datasets and study metadata |
| **KEGG** | Kyoto Encyclopedia of Genes and Genomes | Pathway mapping for microbial functional annotations |
| **PubMed/EuropePMC** | Biomedical literature | Published studies on microbiome-disease associations |
| **CTD** | Comparative Toxicogenomics Database | Chemical-microbiome-disease relationships |

---

## Workflow Overview

```
Phase 0: Query Parsing & Disambiguation
  Identify organism, biome, phenotype, or study accession
    |
Phase 1: Study Discovery
  Search MGnify studies and ENA for relevant datasets
    |
Phase 2: Taxonomic Classification
  Resolve species via GTDB; get lineage and type material
    |
Phase 3: Genome Quality & MAG Assessment
  Search MGnify genomes; evaluate completeness/contamination
    |
Phase 4: Functional Annotation & Pathway Analysis
  MGnify GO/InterPro terms + KEGG pathway mapping
    |
Phase 5: Clinical Phenotype Associations
  GMrepo phenotype-species links; disease relevance
    |
Phase 6: Literature & Disease Context
  PubMed/EuropePMC for published evidence; CTD for gene-disease links
    |
Phase 7: Interpretation & Report Synthesis
  Integrate findings with biological reasoning
```

---

## Phase Details

### Phase 0: Query Parsing & Disambiguation

Parse the user's request to identify:
- **Organism or taxon** (e.g., "Bacteroides fragilis", "Firmicutes")
- **Biome or environment** (e.g., "human gut", "soil", "marine sediment")
- **Phenotype or disease** (e.g., "IBD", "obesity", "colorectal cancer")
- **Study accession** (e.g., MGYS00005292, PRJEB1234)

If the query is ambiguous, clarify before proceeding. A species name might match multiple GTDB entries; a disease might span several biomes.

### Phase 1: Study Discovery

**Objective**: Find relevant metagenomics studies and datasets.

**Tools**:
- `MGnify_search_studies` -- search by keyword, biome, or lineage
  - Input: `query` (search term), optional `biome`, `lineage`
  - Output: list of studies with accession, name, biome, sample count
- `MGnify_search_studies` -- get full details for a known study accession
  - Input: `accession` (MGnify study ID)
  - Output: study metadata, associated analyses, biome hierarchy
- `MGnify_list_biomes` -- browse available biome categories
  - Output: hierarchical biome tree
- `ENAPortal_search_studies` -- search ENA for raw sequencing studies
  - Input: `query` (keyword), optional `result_type`, `limit`
  - **IMPORTANT**: ENA search requires structured queries, not free text. Use field-based queries like `study_title="*IBD*"` or `tax_tree(408170)` for a taxon. Free-text keywords may return HTTP 400.
  - If ENA search fails, fall back to MGnify (which indexes ENA studies with processed results).

**Workflow**:
1. Search MGnify for processed studies matching the query
2. If few MGnify results, broaden with ENA to find raw datasets
3. For each relevant study, note the biome, sequencing type (amplicon vs shotgun), and sample count
4. Retrieve full details for the top studies

**If no studies found**: Try broader search terms, check alternative biome categories via `MGnify_list_biomes`, or search ENA with simpler keywords.

### Phase 2: Taxonomic Classification

**Objective**: Resolve species/genus names to standardized GTDB taxonomy.

**Tools**:
- `GTDB_search_genomes` -- find reference genomes by species name
  - Input: `query` (species name), optional `limit`
  - Output: genome accessions, GTDB taxonomy, quality metrics
- `GTDB_get_species` -- get species cluster details
  - Input: `species_name` (GTDB species name)
  - Output: type genome, member genomes, taxonomy
- `GTDB_get_taxon_info` -- get taxonomic rank details
  - Input: `taxon` (e.g., "g__Bacteroides")
  - Output: rank, parent lineage, child taxa, genome count
- `GTDB_search_taxon` -- search for taxa by partial name
  - Input: `operation="search_taxon"` (REQUIRED), `query` (partial name), optional `rank`
  - Output: matching taxa with lineage

**Workflow**:
1. Search GTDB for the taxon of interest
2. Get full lineage (domain through species)
3. Note any taxonomic reclassifications (GTDB vs NCBI naming differences are common)
4. For species-level queries, identify the type genome and representative genome

**Important**: GTDB uses its own naming conventions (e.g., `s__Bacteroides_A fragilis` may differ from NCBI's `Bacteroides fragilis`). Always note discrepancies.

### Phase 3: Genome Quality & MAG Assessment

**Objective**: Find and evaluate metagenome-assembled genomes (MAGs).

**Tools**:
- `MGnify_search_genomes` -- search the MGnify genome catalog
  - Input: `query` (species/taxon), optional `catalogue`, `limit`
  - Output: genome accessions, completeness, contamination, taxonomy
- `MGnify_get_genome` -- get detailed genome info
  - Input: `accession` (genome ID)
  - Output: full quality metrics (CheckM), gene count, taxonomy, source study

**Workflow**:
1. Search MGnify genomes for the target species
2. Filter by quality: completeness >= 50%, contamination <= 10% (medium quality); completeness >= 90%, contamination <= 5% (high quality)
3. Note the source study and biome for each genome
4. Compare genome statistics (size, GC content, gene count) across entries

**Quality tiers** (MIMAG standard):
- **High quality**: >= 90% complete, <= 5% contamination, 23S/16S/5S rRNA, >= 18 tRNAs
- **Medium quality**: >= 50% complete, <= 10% contamination
- **Low quality**: below medium thresholds -- flag but don't exclude without reason

### Phase 4: Functional Annotation & Pathway Analysis

**Objective**: Extract functional annotations and map them to biological pathways for interpretation.

**Tools**:
- `MGnify_get_go_terms` -- find analysis results for studies
  - Input: `study_accession` or `query`, optional `experiment_type`
  - Output: analysis accessions, pipeline version, experiment type
- `MGnify_get_taxonomy` -- get taxonomic profile from an analysis
  - Input: `accession` (analysis ID)
  - Output: taxonomic composition at multiple ranks
- `MGnify_get_go_terms` -- get GO functional annotations from an analysis
  - Input: `accession` (analysis ID)
  - Output: GO terms with counts
- `kegg_search_pathway` -- search KEGG pathways by keyword
  - Input: `keyword` (NOT `query`; e.g., "butyrate", "lipopolysaccharide")
  - Output: pathway IDs, names, descriptions
- `KEGG_get_pathway_genes` -- get genes in a KEGG pathway
  - Input: `pathway_id` (e.g., "hsa00650" for human butanoate metabolism; use organism prefix like `hsa`, `ko`, or `eco`, NOT bare `map00650`)
  - Output: gene list with functions
- `kegg_get_pathway_info` -- get pathway details
  - Input: `pathway_id` (e.g., "hsa00650")
  - Output: pathway name, description, linked modules

**Workflow**:
1. For studies from Phase 1, retrieve their analyses via `MGnify_get_go_terms`
2. Get GO terms and taxonomic profiles from analyses
3. **Interpret GO terms**: Group by biological process (metabolism, transport, virulence, stress response). Identify which functions are enriched in the study context.
4. **Map to KEGG pathways**: For key functional categories, search KEGG pathways. For example:
   - Short-chain fatty acid production: `kegg_search_pathway(keyword="butanoate")` → hsa00650
   - LPS biosynthesis (inflammation): `kegg_search_pathway(keyword="lipopolysaccharide")` → hsa00540
   - Amino acid metabolism: relevant pathways for tryptophan (hsa00380), bile acid (hsa00120)
5. **Connect function to biology**: Don't just list GO terms. Explain what the functional profile means:
   - Are butyrate producers enriched/depleted? (Relevant to gut barrier integrity)
   - Is LPS biosynthesis capacity high? (Relevant to inflammation)
   - Are bile salt hydrolase genes present? (Relevant to fat metabolism)

**Interpretation framework for common microbiome contexts**:

| Functional Category | Key KEGG Pathways | Biological Significance |
|---|---|---|
| SCFA production | map00650 (butanoate), map00640 (propanoate) | Gut barrier integrity, anti-inflammatory |
| LPS biosynthesis | map00540 | Pro-inflammatory, endotoxemia |
| Bile acid metabolism | map00120 | Fat absorption, FXR signaling |
| Tryptophan metabolism | map00380 | Serotonin, AhR activation, immune |
| Sulfur metabolism | map00920 | H2S production, epithelial damage |
| Vitamin biosynthesis | map00730 (B1), map00740 (B6), map00760 (B12) | Host nutritional contribution |

### Phase 5: Clinical Phenotype Associations

**Objective**: Connect microbial taxa to human health phenotypes.

**Tools**:
- `GMrepo_search_species` -- search for species in the gut microbiome repository
  - Input: `query` (species name or NCBI taxon ID)
  - Output: species prevalence, relative abundance across phenotypes
- `GMrepo_get_phenotypes` -- get phenotypes associated with a species
  - Input: `query` (species name or taxon ID)
  - Output: phenotype list with prevalence and abundance data

**IMPORTANT**: GMrepo uses MeSH-style disease terms, not colloquial names.
- "IBD" → search "Crohn Disease" and "Colitis, Ulcerative" separately
- "obesity" → search "Obesity"
- "diabetes" → search "Diabetes Mellitus, Type 2"
- "colorectal cancer" → search "Colorectal Neoplasms"
If a search returns 0 results, try the formal MeSH term or the NCBI taxon ID instead.

**Workflow**:
1. For key taxa identified in earlier phases, query GMrepo
2. Get phenotype associations -- which diseases/conditions show enrichment or depletion
3. Note the sample sizes and prevalence ranges
4. Cross-reference with study findings from Phase 1

**Scope note**: GMrepo focuses on the human gut microbiome. For environmental metagenomics (soil, marine), this phase may be limited or skippable.

### Phase 6: Literature & Disease Context

**Objective**: Ground findings in published evidence and connect to disease mechanisms.

This phase is critical -- it transforms data collection into scientific interpretation.

**Tools**:
- `PubMed_search_articles` -- search biomedical literature
  - Input: `query` (search terms), `max_results`
  - Strategy: Search "[taxon] AND [disease]" and "[taxon] AND microbiome AND [condition]"
- `EuropePMC_search_articles` -- broader literature including preprints
  - Input: `query`, `page_size`
  - Useful for recent/European studies not yet in PubMed
- `DisGeNET_search_gene` -- gene-disease associations (for microbial gene products affecting host)
  - Input: `query` (gene symbol)
- `CTD_get_gene_diseases` -- gene-disease associations from CTD
  - Input: `input_terms` (gene symbol or NCBI Gene ID, e.g., "IL10")
  - Useful for: connecting microbial-influenced host genes to disease

**Workflow**:
1. For each key taxon-disease association from Phase 5, search PubMed for published evidence
2. Look for systematic reviews and meta-analyses first (highest evidence)
3. Search for mechanistic studies (how does this microbe contribute to disease?)
4. Use CTD to find metabolite-disease connections (e.g., "butyrate" → protective in IBD, "trimethylamine" → cardiovascular risk)
5. Synthesize: Is the association well-established (multiple studies, consistent direction) or preliminary (single study, conflicting)?

**Evidence grading for microbiome associations**:

| Grade | Criteria | Example |
|---|---|---|
| **Strong** | Meta-analysis or >5 independent studies, consistent direction | F. prausnitzii depletion in Crohn's disease |
| **Moderate** | 2-5 studies with consistent direction, or 1 large cohort study | Akkermansia muciniphila and metabolic health |
| **Preliminary** | Single study or conflicting results | Novel species-disease link from one cohort |
| **Mechanistic only** | In vitro or animal model data, no human epidemiology | Specific metabolite pathway demonstrated in gnotobiotic mice |

### Phase 7: Interpretation & Report Synthesis

**Objective**: Synthesize all findings into a scientifically meaningful narrative, not just a data dump.

This is what separates a useful analysis from a tool catalog. After collecting data from Phases 1-6, answer these questions:

**For disease-microbiome studies**:
1. **What changed?** Which taxa are enriched/depleted in disease vs control? (From Phase 2 + Phase 5)
2. **Why does it matter?** What functional consequences follow from the compositional shift? (From Phase 4)
   - e.g., "Depletion of F. prausnitzii reduces butyrate production → weakened gut barrier → increased intestinal permeability"
3. **Is it cause or consequence?** Does the literature support a causal role or just correlation? (From Phase 6)
4. **What's the mechanism?** Connect microbial functions to host pathways (Phase 4 + Phase 6)
5. **How confident are we?** Grade the overall evidence (Phase 6 evidence grading)

**For taxonomic/ecological studies**:
1. **What's the community structure?** Dominant taxa, diversity, evenness
2. **How does it compare to reference communities?** (GMrepo healthy baseline)
3. **What functional capacity does the community have?** (KEGG pathway mapping)
4. **Are there indicator species?** Taxa strongly associated with the biome or condition

**Report structure**:

1. **Executive Summary** -- One paragraph answering the user's question directly
2. **Study Landscape** -- number of studies found, biomes covered, sequencing methods
3. **Taxonomic Profile** -- GTDB-standardized taxonomy, key lineages, naming discrepancies
4. **Functional Interpretation** -- pathway analysis results with biological significance (not just GO term lists)
5. **Clinical Relevance** -- phenotype associations with evidence grading
6. **Mechanistic Model** -- How do the identified taxa contribute to the condition? (diagram or narrative)
7. **Genome Catalog** -- available MAGs with quality tiers, representative genomes
8. **Data Gaps** -- missing biomes, low-quality genomes, taxa without clinical data, conflicting evidence

---

## Common Analysis Patterns

| Pattern | Description | Key Phases |
|---------|-------------|------------|
| **Species Deep-Dive** | Full profile of one species: taxonomy, genomes, clinical links, literature | 0, 2, 3, 5, 6 |
| **Study Exploration** | Discover and summarize studies for a biome or disease | 0, 1, 4, 6 |
| **MAG Quality Audit** | Evaluate genome quality for a set of species | 0, 2, 3 |
| **Disease-Microbiome Link** | Find taxa associated with a clinical condition + mechanisms | 0, 1, 4, 5, 6, 7 |
| **Taxonomic Reconciliation** | Compare GTDB vs NCBI classification for a lineage | 0, 2 |
| **Functional Profiling** | What metabolic capacities exist in a community? | 0, 1, 4, 7 |

---

## Edge Cases & Fallbacks

- **Taxon not in GTDB**: May be a recently described species or use outdated naming. Try partial search via `GTDB_search_taxon`, or fall back to MGnify which uses NCBI taxonomy
- **No GMrepo data**: Normal for non-gut organisms. Note the gap and use literature search (Phase 6) instead
- **MGnify study vs ENA study**: MGnify contains processed results; ENA has raw data. If MGnify lacks a study, ENA may still have the raw reads
- **ENA search returns HTTP 400**: Use structured field queries, not free text. Try `MGnify_search_studies` as fallback
- **GMrepo returns 0 results for disease term**: Use formal MeSH terms (e.g., "Crohn Disease" not "IBD", "Colitis, Ulcerative" not "UC"). Try NCBI taxon IDs for species
- **Large result sets**: MGnify and ENA can return hundreds of studies. Use biome filters and limit parameters to focus results
- **GTDB version differences**: Taxonomy changes between GTDB releases. Note the version when reporting classifications
- **No KEGG pathway match**: Some microbial functions are poorly mapped in KEGG. Check MetaCyc or literature for pathway information

---

## Limitations

- **GMrepo**: Gut-only; no coverage for skin, oral, vaginal, or environmental microbiomes
- **GTDB**: Bacteria and Archaea only; no eukaryotic microbes or viruses
- **MGnify**: Functional annotations depend on pipeline version; older analyses may lack newer annotations
- **ENA**: Raw data only; no processed taxonomic profiles. Query syntax is strict (not free-text friendly)
- **No direct sequence analysis**: This skill queries databases, not raw FASTQ/FASTA files. For sequence processing, use external pipelines (QIIME2, MetaPhlAn) and bring results here for annotation
- **Statistical comparison**: For differential abundance between groups, use the computational procedure below. For full DESeq2/ANCOM pipelines on raw count data, process externally and bring results here for annotation.

### Computational Procedure: Differential Abundance Analysis

When comparing taxa between conditions (e.g., IBD vs healthy), use this procedure with taxonomic profiles from MGnify or user-provided data:

```python
# Differential abundance analysis using scipy + pandas
# Requires: pandas, scipy, numpy (all in ToolUniverse dependencies)
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from scipy.stats import false_discovery_control

# Input: abundance table (rows=samples, cols=taxa, values=relative abundance)
# Columns: taxa names; last column or separate metadata has group labels
# Example with MGnify data:
abundance = pd.DataFrame({
    'Faecalibacterium': [0.12, 0.15, 0.08, 0.02, 0.01, 0.03],
    'Bacteroides': [0.20, 0.18, 0.22, 0.30, 0.28, 0.25],
    'Akkermansia': [0.05, 0.03, 0.04, 0.01, 0.00, 0.02],
    'group': ['healthy', 'healthy', 'healthy', 'IBD', 'IBD', 'IBD']
})

taxa_cols = [c for c in abundance.columns if c != 'group']
results = []
for taxon in taxa_cols:
    healthy = abundance[abundance['group'] == 'healthy'][taxon]
    disease = abundance[abundance['group'] == 'IBD'][taxon]
    stat, pval = mannwhitneyu(healthy, disease, alternative='two-sided')
    log2fc = np.log2((disease.mean() + 1e-6) / (healthy.mean() + 1e-6))
    results.append({
        'taxon': taxon,
        'healthy_mean': healthy.mean(),
        'disease_mean': disease.mean(),
        'log2FC': log2fc,
        'direction': 'depleted' if log2fc < 0 else 'enriched',
        'pvalue': pval
    })

res_df = pd.DataFrame(results)
# FDR correction
res_df['padj'] = false_discovery_control(res_df['pvalue'], method='bh')
res_df = res_df.sort_values('pvalue')
print(res_df.to_string(index=False))

# Interpretation:
# - Depleted + padj < 0.05: taxon significantly reduced in disease
# - Enriched + padj < 0.05: taxon significantly increased in disease
# - |log2FC| > 1: biologically meaningful effect size
```

**When to use this**: After collecting taxonomic profiles from MGnify (Phase 4) or user-provided 16S/shotgun data. This replaces external tools like LEfSe for simple two-group comparisons. For multi-group or longitudinal designs, recommend ANCOM-BC2 or MaAsLin2 externally
- **Pathway mapping is approximate**: KEGG pathways represent reference pathways, not community-specific reconstructions. Use as interpretive framework, not definitive functional prediction
