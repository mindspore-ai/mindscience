---
name: tooluniverse-literature-deep-research
description: Comprehensive literature deep research across any academic domain using 120+ ToolUniverse tools. Conducts subject disambiguation, systematic literature search with citation network expansion, evidence grading (T1-T4), and structured theme extraction. Produces detailed reports with mandatory completeness checklists, integrated models, and testable hypotheses. Use when users need thorough literature reviews, target/drug/disease profiles, topic deep-dives, claim verification, or systematic evidence synthesis. Supports biomedical (genes, proteins, drugs, diseases), computer science, social science, and general academic topics. For single factoid questions, uses a fast verification mode with inline answer.

license: MIT License
metadata:
    skill-author: ToolUniverse (https://github.com/mims-harvard/TxAgent)
---

# Literature Deep Research

Systematic approach to comprehensive literature research: disambiguate the subject, search with collision-aware queries, grade evidence, and produce a structured report.

**KEY PRINCIPLES**:
1. **Disambiguate first** - Resolve IDs, synonyms, naming collisions before literature search
2. **Right-size the deliverable** - Factoid mode for single questions; full report for deep research
3. **Evidence grading** - Grade every claim (T1 mechanistic → T4 mention)
4. **Mandatory completeness** - All sections must exist, even if "unknown/limited evidence"
5. **Source attribution** - Every claim traceable to database/tool
6. **English-first queries** - Use English for searches; respond in user's language
7. **Report = deliverable** - Show findings, not search process

---

## Workflow Overview

```
User Query
  ↓
Phase 0: CLARIFY + MODE SELECT (factoid vs deep report)
  ↓
Phase 1: SUBJECT DISAMBIGUATION + PROFILE
  ├─ Detect domain (biological target / drug / disease / general academic)
  ├─ Resolve identifiers and gather synonyms/aliases
  ├─ Check for naming collisions
  └─ Gather baseline context via annotation tools (domain-specific)
  ↓
Phase 2: LITERATURE SEARCH (methodology kept internal)
  ├─ High-precision seed queries
  ├─ Citation network expansion from seeds
  ├─ Collision-filtered broader queries
  └─ Theme clustering + evidence grading
  ↓
Phase 3: REPORT SYNTHESIS (report-first pattern)
  ├─ Create [topic]_report.md with all section headers IMMEDIATELY
  ├─ Progressively fill sections as data arrives (update after each phase)
  ├─ Write Executive Summary LAST (after all sections complete)
  ├─ Generate [topic]_bibliography.json + .csv
  └─ Validate completeness checklist
```

---

## Phase 0: Initial Clarification

Ask only what is needed; skip questions with obvious answers:

1. **Subject type**: Gene/protein, disease, drug, CS/ML topic, social science, or general?
2. **Scope**: Single factoid to verify, or comprehensive deep review?
3. **Known aliases** (if ambiguous): Specific names or symbols in use?
4. **Constraints**: Open access only? Include preprints? Specific organisms or date range?

### Mode Selection

| Mode | When to Use | Deliverable |
|------|-------------|-------------|
| **Factoid / Verification** | Single concrete question | `[topic]_factcheck_report.md` (≤1 page) + bibliography |
| **Mini-review** | Narrow topic | Short narrative report (1-3 pages) |
| **Full Deep-Research** | Comprehensive overview | Full 15-section report + bibliography |

**Heuristic**: "Which antibiotic was X evolved to resist?" → Factoid. "What does the literature say about X?" → Full.

### Factoid / Verification Mode (Fast Path)

Provide a correct, source-verified answer with explicit evidence attribution.

```markdown
# [TOPIC]: Fact-check Report
*Generated: [Date]*

## Question
[User question]

## Answer
**[One-sentence answer]** [Evidence: ★★★/★★☆/★☆☆/☆☆☆]

## Source(s)
- [Primary citation: journal/year/PMID/DOI]

## Verification Notes
- [1-3 bullets: where the statement appears, key constraints]

## Limitations
- [Full text availability, evidence type caveats]
```

Prefer ToolUniverse literature tools over web browsing. Use `EuropePMC_search_articles(extract_terms_from_fulltext=[...])` for OA snippet verification when possible.

### Detect Subject Domain

| Query Pattern | Domain | Phase 1 Action |
|---------------|--------|----------------|
| Gene symbol (EGFR, TP53) | Biological target | Full bio disambiguation |
| Protein name ("V-ATPase") | Biological target | Full bio disambiguation |
| Drug name ("metformin") | Drug | Drug disambiguation (see 1.5) |
| Disease ("Alzheimer's") | Disease | Disease disambiguation (see 1.6) |
| CS/ML topic ("transformer architecture") | General academic | Literature-only (skip bio tools) |
| Method, concept, general topic | General academic | Literature-only (skip bio tools) |
| Cross-domain ("GNNs for drug discovery") | Interdisciplinary | Resolve each entity in its domain (see 1.9) |

### Cross-Skill Delegation

For deep entity-specific research beyond literature, delegate to specialized skills:
- **Gene/protein deep-dive** (9-path profiling, druggability, GPCR data): use `tooluniverse-target-research`
- **Drug comprehensive profile** (ADMET, FDA labels, formulations): use `tooluniverse-drug-research`
- **Disease comprehensive profile** (ontologies, epidemiology, treatments): use `tooluniverse-disease-research`

Use this skill when the focus is **literature synthesis and evidence grading**. Use specialized skills when the focus is **entity profiling with structured database queries**. For maximum depth, run both in parallel.

---

## Phase 1: Subject Disambiguation + Profile

### 1.1 Resolve Official Identifiers (Biological Targets)

```
UniProt_search → UniProt accession
UniProt_get_entry_by_accession → Full entry with cross-references
UniProt_id_mapping → Map between ID types
ensembl_lookup_gene → Ensembl gene ID, biotype
MyGene_get_gene_annotation → NCBI Gene ID, aliases, summary
```

### 1.2 Naming Collision Detection

Check the primary database for the domain (first 20 results). If >20% off-topic, build a negative filter:

| Domain | Collision Check Syntax |
|--------|----------------------|
| Biomedical | PubMed: `"[TERM]"[Title]` |
| CS/ML | ArXiv: `ti:"[TERM]"` or SemanticScholar with `fieldsOfStudy` filter |
| General | OpenAlex or Crossref title search |

1. Identify collision terms from off-topic results
2. Build negative filter: `NOT [collision1] NOT [collision2]`

**Gene family disambiguation**: Use official symbol with explicit exclusions.
Example: `"ADAR" NOT "ADAR2" NOT "ADARB1"` for ADAR1-specific results.

**Cross-domain collision**: Some terms have different meanings across fields (e.g., "RAG" = Retrieval-Augmented Generation in CS, Recombination Activating Gene in biology). Add domain context terms to filter: `"RAG" AND "language model" NOT "recombination activating"`.

### 1.3 Baseline Profile (Biological Targets)

Gather structural, functional, and expression context via annotation tools:

```
InterPro_get_protein_domains → Domain architecture
UniProt_get_ptm_processing_by_accession → PTMs, active sites
HPA_get_subcellular_location → Localization
GTEx_get_median_gene_expression → Tissue expression (use gtex_v8)
GO_get_annotations_for_gene → GO terms
Reactome_map_uniprot_to_pathways → Pathways
STRING_get_protein_interactions → Interaction partners
intact_get_interactions → Experimentally validated PPIs
OpenTargets_get_target_tractability_by_ensemblID → Druggability assessment
```

**GPCR targets**: If the target is a GPCR (~35% of approved drug targets), delegate to `tooluniverse-target-research` for specialized GPCRdb data (3D structures, ligands, mutations).

### 1.4 Baseline Profile Output

```markdown
## Target Identity
| Identifier | Value | Source |
|------------|-------|--------|
| Official Symbol | [SYMBOL] | HGNC |
| UniProt | [ACC] | UniProt |
| Ensembl Gene | [ENSG...] | Ensembl |

**Synonyms**: [list]
**Collisions**: [assessment]
```

### 1.5 Drug-Centric Disambiguation

Skip protein architecture/expression/GO. Instead:

**Resolve identity**: `OpenTargets_get_drug_chembId_by_generic_name`, `ChEMBL_get_drug`, `PubChem_get_CID_by_compound_name`, `drugbank_get_drug_basic_info_by_drug_name_or_id`

**Targets & mechanisms**: `ChEMBL_get_drug_mechanisms`, `OpenTargets_get_associated_targets_by_drug_chemblId`, `DGIdb_get_drug_gene_interactions`, `drugbank_get_targets_by_drug_name_or_drugbank_id`

**Safety & indications**: `OpenTargets_get_drug_adverse_events_by_chemblId`, `OpenTargets_get_drug_indications_by_chemblId`, `search_clinical_trials`

### 1.6 Disease-Centric Disambiguation

**Resolve ontology IDs**: Use `OpenTargets_get_drug_chembId_by_generic_name` or disease search tools to resolve EFO/MONDO IDs. Cross-reference ICD-10 and UMLS CUI when available from tool results.

```
OpenTargets_get_diseases_phenotypes_by_target_ensembl → Disease associations
DisGeNET_get_disease_genes → Disease-gene associations
DisGeNET_search_disease → Disease search with ontology IDs
CTD_get_disease_chemicals → Chemical-disease links
```

### 1.7 Compound Queries (e.g., "metformin in breast cancer")

Resolve both entities separately, then cross-reference:
```
CTD_get_chemical_gene_interactions → Chemical-gene links
CTD_get_chemical_diseases → Chemical-disease associations
OpenTargets_get_associated_targets_by_drug_chemblId → Drug targets
OpenTargets_get_associated_diseases_by_drug_chemblId → Drug-disease associations
→ Intersect to find shared targets/pathways
```

### 1.8 General Academic Topics (No Bio Tools)

For CS, social science, humanities, or other non-bio topics:
- Skip all bio annotation tools (UniProt, InterPro, GTEx, etc.)
- Proceed directly to Phase 2 literature search
- Use domain-appropriate databases (ArXiv for CS/ML, DBLP for CS, OSF for social science)
- Collision detection still applies (search term ambiguity)

### 1.9 Interdisciplinary / Cross-Domain Queries

For topics spanning multiple domains (e.g., "GNNs for drug discovery", "AlphaFold protein prediction"):
1. **Identify each domain component** separately (e.g., CS method + biological application)
2. **Resolve bio entities** using Phase 1.1-1.3 (targets, drugs, diseases)
3. **Search CS/general literature** using ArXiv, DBLP, SemanticScholar in parallel
4. **Merge results** — use both bio tools AND general academic tools in Phase 2
5. **Cross-reference** — find papers that bridge both domains (typically computational biology venues)

---

## Phase 2: Literature Search

**Methodology stays internal. The report shows findings, not process.**

### 2.1 Query Strategy

**Step 1: High-Precision Seeds** (15-30 core papers)

Domain-specific seed queries:
```
Biomedical: "[TERM]"[Title] AND (mechanism OR function OR structure OR review)
CS/ML:      ti:"[TERM]" AND (architecture OR benchmark OR evaluation OR survey)
General:    "[TERM]" in title via OpenAlex/Crossref
```

Use date/sort filters for recency or impact:
- PubMed: `mindate`, `maxdate`, `sort="pub_date"`
- SemanticScholar: `year="2023-2024"`, `sort="citationCount:desc"`
- ArXiv: `date_from`, `sort_by="submittedDate"`

**Step 2: Citation Network Expansion**
```
PubMed_get_cited_by → Forward citations (primary)
EuropePMC_get_citations → Forward (fallback)
PubMed_get_related → Related papers
EuropePMC_get_references → Backward citations
SemanticScholar_get_recommendations → AI-similar papers
OpenCitations_get_citations → DOI-based citation data
```

**Step 3: Collision-Filtered Broader Queries**
```
"[TERM]" AND ([context1] OR [context2]) NOT [collision_term]
```

### 2.2 Literature Search Tools

**Biomedical**: `PubMed_search_articles`, `PMC_search_papers`, `EuropePMC_search_articles`, `PubTator3_LiteratureSearch`

**CS/ML**: `ArXiv_search_papers`, `DBLP_search_publications`, `SemanticScholar_search_papers`

**General academic**: `openalex_literature_search`, `Crossref_search_works`, `CORE_search_papers`, `DOAJ_search_articles`

**Preprints**: `BioRxiv_get_preprint`, `MedRxiv_get_preprint`, `OSF_search_preprints`, `BioRxiv_list_recent_preprints`
(For preprint keyword search: `EuropePMC_search_articles(source='PPR')`)

**Multi-source deep search**: `advanced_literature_search_agent` (searches 12+ databases in parallel; requires Azure OpenAI key — if unavailable, replicate coverage by querying PubMed + ArXiv + SemanticScholar + OpenAlex individually)

**Citation impact**: `iCite_search_publications` (search + RCR/APT metrics), `iCite_get_publications` (metrics by PMID), `scite_get_tallies` (supporting/contradicting counts)
*Note: iCite and scite are PubMed-only. For CS/ML papers, use `SemanticScholar_get_paper` for citation counts and influence scores.*

**Author search**: PubMed `"Author[Author]"`, ArXiv `"au:Name"`, SemanticScholar/OpenAlex as query text

### 2.3 Full-Text Verification

When abstracts lack critical details, use full-text snippet extraction. See `FULLTEXT_STRATEGY.md` for the three-tier strategy (Europe PMC auto-snippets → manual Semantic Scholar/ArXiv → manual download).

### 2.4 Tool Failure Handling

```
Attempt 1 → fails → wait 2s → Attempt 2 → fails → wait 5s → Fallback tool
```

| Primary | Fallback 1 | Fallback 2 |
|---------|------------|------------|
| `PubMed_get_cited_by` | `EuropePMC_get_citations` | `OpenCitations_get_citations` |
| `PubMed_get_related` | `SemanticScholar_get_recommendations` | `SemanticScholar_search_papers` |
| `GTEx_get_median_gene_expression` | `HPA_get_rna_expression_by_source` | Document as unavailable |
| `Unpaywall_check_oa_status` | Europe PMC `isOpenAccess` | OpenAlex `is_oa` |

### 2.5 Open Access Handling

With Unpaywall email: full OA check. Without: best-effort via Europe PMC, PMC, OpenAlex, DOAJ flags.
Label: `*OA Status: Best-effort (Unpaywall not configured)*`

---

## Phase 3: Evidence Grading

Grade every claim by evidence strength:

| Tier | Label | Description | Bio Example | CS/ML Example |
|------|-------|-------------|-------------|---------------|
| **T1** | ★★★ Mechanistic | Direct experimental/formal evidence | CRISPR KO + rescue, RCT | Formal proof, controlled ablation with significance test |
| **T2** | ★★☆ Functional | Functional study showing role | siRNA knockdown phenotype | Benchmark on standard dataset with baselines |
| **T3** | ★☆☆ Association | Screen hit, correlation, observational | High-throughput screen, GWAS | Observational study, case study, anecdotal comparison |
| **T4** | ☆☆☆ Mention | Review, text-mined, peripheral | Review article | Survey paper, blog post, workshop abstract |

**In report**, label inline:
```markdown
Target X regulates pathway Y [★★★: PMID:12345678] through direct
phosphorylation [★★☆: PMID:23456789].
```

**Per theme**, summarize evidence quality:
```markdown
### Theme: Lysosomal Function (47 papers)
**Evidence Quality**: Strong (32 mechanistic, 11 functional, 4 association)
```

---

## Report Output

### Deliverables

| File | Mode | Always? |
|------|------|---------|
| `[topic]_report.md` | Full Deep-Research | Yes |
| `[topic]_factcheck_report.md` | Factoid | Yes |
| `[topic]_bibliography.json` | All modes | Yes |
| `[topic]_bibliography.csv` | All modes | Yes |
| `methods_appendix.md` | Any (only if requested) | No |

### Report-First Progressive Update Pattern

**Create the report file immediately** after Phase 0 with all 15 section headers (use template from `REPORT_TEMPLATE.md`). Then:
1. After Phase 1 (disambiguation): fill Sections 1-5
2. After Phase 2 (literature search): fill Sections 6-12
3. After evidence grading: fill Sections 13-14
4. **Last**: write Executive Summary and Section 15 (synthesizes everything)

This ensures partial results are saved even if the process is interrupted.

### Report Template

Use the 15-section template from `REPORT_TEMPLATE.md`. Key sections adapt by domain:
- **Biological targets**: protein architecture, expression, GO terms, disease links, pathogen involvement
- **Drugs**: chemical properties, targets/MOA, pharmacokinetics, indications, safety
- **Diseases**: epidemiology, pathophysiology, associated genes, treatments
- **General academic**: historical context, key theories, empirical evidence, applications

See `REPORT_TEMPLATE.md` for full template, domain-specific adaptations, bibliography format, theme extraction protocol, and completeness checklist.

---

## Communication

**Brief progress updates** (not search logs):
- "Resolving subject identifiers..."
- "Building core paper set..."
- "Expanding via citation network..."
- "Clustering themes and grading evidence..."

**DO NOT expose**: raw tool outputs, dedup counts, search round details, database-by-database results.

**For factoid queries**: ask (once) if user wants just the verified answer or a full report. Default to factoid mode.

---

## References

- **`TOOL_NAMES_REFERENCE.md`** — Complete list of 123 tools with parameters
- **`REPORT_TEMPLATE.md`** — Full report template, domain adaptations, bibliography format, theme extraction, completeness checklist
- **`FULLTEXT_STRATEGY.md`** — Three-tier full-text verification strategy
- **`WORKFLOW.md`** — Compact workflow cheat-sheet
- **`EXAMPLES.md`** — Worked examples (ATP6V1A, TRAG collision, sparse target, drug query)
