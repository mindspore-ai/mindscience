---
name: tooluniverse-acmg-variant-classification
description: Systematic ACMG/AMP variant classification using ToolUniverse tools. Given a genetic variant (HGVS, rsID, or gene+change), applies all 28 ACMG criteria (PVS1, PS1-4, PM1-6, PP1-5, BA1, BS1-4, BP1-7) through automated database queries and computational predictions. Produces a final 5-tier classification (Pathogenic / Likely Pathogenic / VUS / Likely Benign / Benign) with evidence summary. Use when asked to classify a variant, interpret a VUS, apply ACMG criteria, assess pathogenicity, or determine clinical significance of a germline variant.

license: MIT License
metadata:
    skill-author: ToolUniverse (https://github.com/mims-harvard/TxAgent)
---

# ACMG/AMP Variant Classification

Systematic application of the 28 ACMG/AMP criteria to classify germline variants into five tiers: Pathogenic, Likely Pathogenic, VUS, Likely Benign, or Benign. Each phase queries specific databases, then the classification algorithm combines criteria per Richards et al., 2015.

**KEY PRINCIPLES**:
1. **Criteria-driven** - Every classification must cite which ACMG criteria were activated and why
2. **Conservative** - When evidence is ambiguous, do NOT upgrade a criterion; leave it unmet
3. **Gene-aware** - Adjust thresholds based on gene mechanism (LOF vs. gain-of-function)
4. **Population-calibrated** - Use ancestry-specific gnomAD frequencies, not just global AF
5. **Transparent** - Show evidence for each criterion so clinicians can audit the reasoning
6. **Source-referenced** - Every criterion activation must cite the database/tool source
7. **English-first queries** - Always use English terms in tool calls. Respond in user's language

---

## When to Use

- "Classify BRCA2 c.5946delT using ACMG criteria"
- "Is this VUS pathogenic? NM_000059.4:c.7397T>C"
- "Apply ACMG guidelines to rs28897743"
- "What is the pathogenicity of CFTR p.Arg117His?"
- "ACMG classification for TP53 R248W"

---

## Tool Parameter Reference (CRITICAL)

| Tool | Key Parameters | Notes |
|------|---------------|-------|
| `VariantValidator_validate_variant` | `variant_description`, `genome_build`, `select_transcripts` | genome_build="GRCh38" |
| `VariantValidator_gene2transcripts` | `gene_symbol` | Returns MANE Select transcript |
| `MyVariant_query_variants` | `query` | HGVS or rsID. Returns ClinVar, gnomAD, CADD, REVEL, SIFT, PolyPhen |
| `EnsemblVEP_annotate_hgvs` | `hgvs_notation` | Consequence, colocated variants, ancestry gnomAD |
| `gnomad_search_variants` | `query` | rsID to gnomAD variant ID |
| `gnomad_get_variant` | `variant_id` | Per-ancestry population frequencies |
| `gnomad_get_gene_constraints` | `gene_symbol` | pLI, LOEUF, mis_z |
| `ClinVar_search_variants` | `query` | Variable response format: list OR `{status, data}` |
| `ClinVar_get_variant_details` | `variant_id` | ClinVar numeric ID |
| `civic_get_variants_by_gene` | `gene_id` | CIViC numeric gene ID (NOT symbol). Known: BRAF=5, BRCA2=19 |
| `UniProt_get_function_by_accession` | `accession` | Returns list of strings |
| `InterPro_get_entries_for_protein` | `accession` | Domain architecture by UniProt accession |
| `alphafold_get_prediction` | `qualifier` | UniProt accession; pLDDT confidence |
| `PubMed_search_articles` | `query`, `limit` | Returns list of dicts |
| `MyGene_query_genes` | `query` | Filter by `symbol` match (first hit may not match) |

---

## Phase 0: Variant Validation and Normalization

**WHY**: ACMG classification requires an unambiguous variant on a specific transcript. Wrong HGVS or transcript cascades errors through all criteria. MANE Select is the community-agreed reference.

1. **Get MANE Select transcript**: `VariantValidator_gene2transcripts(gene_symbol="BRCA2")`
2. **Validate variant**: `VariantValidator_validate_variant(variant_description="NM_000059.4:c.5946delT", genome_build="GRCh38", select_transcripts="mane_select")`
3. **Resolve gene IDs**: `MyGene_query_genes(query="BRCA2")` -- extract Ensembl ID, UniProt accession. Filter by `symbol == 'BRCA2'`.
4. **Record**: HGVS coding, HGVS protein, genomic coordinates, variant type (frameshift/missense/nonsense/splice/synonymous/in-frame indel).

**Accepted inputs**: HGVS coding (NM_000059.4:c.5946delT), HGVS protein (BRCA2 p.Val600Glu), rsID (rs28897743), gene+change (BRCA1 c.68_69del), genomic coordinates.

---

## Phase 1: Population Frequency Data

**WHY**: Population AF is among the strongest evidence. A variant at >5% in any population is almost certainly benign (BA1). Absent from gnomAD supports pathogenicity (PM2). Ancestry-specific frequencies prevent misclassifying population-enriched benign variants.

### Criteria: BA1, BS1, BS2, PM2

1. **gnomAD frequencies**: `gnomad_search_variants(query="rs28897743")` then `gnomad_get_variant(variant_id=...)` for per-ancestry AF.
2. **MyVariant fallback**: `MyVariant_query_variants(query="rs28897743")` -- access `hit['gnomad_genome']['af']`.
3. **Gene constraints**: `gnomad_get_gene_constraints(gene_symbol="BRCA2")` -- pLI, LOEUF, mis_z.

| Criterion | Strength | Condition |
|-----------|----------|-----------|
| **BA1** | Stand-alone benign | AF >= 0.05 in ANY ancestry population |
| **BS1** | Strong benign | AF above disease-specific cutoff (default 0.01 common, 0.001 rare) |
| **BS2** | Strong benign | Observed in healthy homozygotes (recessive) or healthy adults (dominant with full penetrance) |
| **PM2** | Supporting path | Absent or AF < 0.0001 in gnomAD. ClinGen recommends PM2_Supporting for most genes |

**Nuances**: Use ancestry-specific AF, not just global. For BS1, threshold = `prevalence x max_allelic_contribution x max_genetic_contribution / penetrance`. If gnomAD unavailable, note gap and continue.

---

## Phase 2: Computational Predictions

**WHY**: No single predictor suffices, but concordance across REVEL, CADD, AlphaMissense, SIFT, PolyPhen provides supporting evidence. Applies only to missense variants.

### Criteria: PP3, BP4

1. **MyVariant predictions**: `MyVariant_query_variants(query="...")` -- extract `cadd.phred`, `dbnsfp.revel_score`, `dbnsfp.alphamissense`, `cadd.sift`, `cadd.polyphen`.
2. **VEP annotation**: `EnsemblVEP_annotate_hgvs(hgvs_notation="...")` -- consequence_terms, SIFT/PolyPhen, SpliceAI deltas.

| Predictor | Damaging Threshold | Benign Threshold |
|-----------|-------------------|-----------------|
| REVEL | >= 0.7 | < 0.15 |
| CADD PHRED | >= 25 | < 15 |
| AlphaMissense | >= 0.564 | < 0.34 |
| SIFT | < 0.05 | >= 0.05 |
| PolyPhen | >= 0.85 | < 0.15 |

| Criterion | Condition |
|-----------|-----------|
| **PP3** (Supporting path) | Majority predict damaging (>= 3/5 concordant). REVEL >= 0.7 alone suffices per ClinGen |
| **BP4** (Supporting benign) | ALL predict benign/tolerated. REVEL < 0.15 or CADD < 15 |

**Nuances**: Only for missense. Discordant predictions = neither PP3 nor BP4. For non-missense, use SpliceAI (Phase 5).

---

## Phase 3: Clinical Database Evidence

**WHY**: ClinVar aggregates clinical lab classifications. Same amino acid change from different nucleotide (PS1) or different pathogenic missense at same residue (PM5) are strong/moderate evidence.

### Criteria: PS1, PM5, PP5, BP6

1. **ClinVar**: `ClinVar_search_variants(query="BRCA2 c.5946delT")` then `ClinVar_get_variant_details(variant_id=...)`.
2. **CIViC**: `civic_get_variants_by_gene(gene_id=19)` -- check same variant and same-residue variants.

| Criterion | Condition |
|-----------|-----------|
| **PS1** (Strong path) | Same amino acid change as established pathogenic variant from DIFFERENT nucleotide change. Verify mechanism is amino acid (not splicing) |
| **PM5** (Moderate path) | Different pathogenic missense at SAME residue (e.g., Arg248Trp pathogenic -> Arg248Gln gets PM5) |
| **PP5** (Supporting path) | ClinVar Pathogenic with >= 2-star review. Weight by concordant submitter count |
| **BP6** (Supporting benign) | ClinVar Benign/Likely Benign with concordant submitters |

**Nuances**: Conflicting ClinVar interpretations = do NOT apply PP5/BP6. ClinGen has proposed downweighting PP5/BP6.

---

## Phase 4: Functional Domain and Protein Analysis

**WHY**: Variants in well-established functional domains with pathogenic variant enrichment are more likely pathogenic. Structural data helps assess variant impact.

### Criteria: PM1, PP2, BP1

1. **Protein function**: `UniProt_get_function_by_accession(accession="P51587")` -- active sites, binding sites.
2. **Domain architecture**: `InterPro_get_entries_for_protein(accession="P51587")` -- map variant position to domains.
3. **Structural context** (optional): `alphafold_get_prediction(qualifier="P51587")` -- high pLDDT (>90) = well-structured region.

| Criterion | Condition |
|-----------|-----------|
| **PM1** (Moderate path) | In well-established functional domain AND domain is mutational hotspot with low benign variation |
| **PP2** (Supporting path) | Missense in gene with low benign missense rate (mis_z > 3.09) where missense is known mechanism |
| **BP1** (Supporting benign) | Missense in gene where ONLY truncating variants cause disease (LOF-only mechanism) |

**PP2 and BP1 are mutually exclusive.**

---

## Phase 5: Splice Impact Assessment

**WHY**: Splice-disrupting variants can cause exon skipping/intron retention leading to frameshifts. PVS1 is the strongest single pathogenic criterion.

### Criteria: PVS1

1. **VEP consequence** (from Phase 2): look for `splice_donor_variant`, `splice_acceptor_variant`.
2. **SpliceAI** (from VEP/MyVariant): delta >= 0.5 = strong; 0.2-0.5 = moderate; < 0.2 = unlikely.
3. **Gene LOF mechanism**: pLI >= 0.9 OR LOEUF < 0.35 supports LOF intolerance.

**PVS1 Decision Framework**:
- Null variant (nonsense/frameshift/canonical splice/initiation codon) + LOF is known mechanism + not in last exon/last 50bp penultimate exon + no rescue transcript = **PVS1** (full)
- Last exon / NMD escape likely = **PVS1_Moderate**
- Rescue transcript possible = **PVS1_Supporting**
- SpliceAI >= 0.5 but not canonical site = **PVS1_Supporting**
- LOF mechanism uncertain = do NOT apply PVS1

---

## Phase 6: Literature and Functional Evidence

**WHY**: Published functional studies provide strong evidence. Well-designed assays showing LOF (PS3) or normal function (BS3) can shift classification decisively.

### Criteria: PS3, BS3, PP1, PP4

1. **Functional studies**: `PubMed_search_articles(query="BRCA2 c.5946delT functional assay", limit=10)`
2. **Segregation data**: `PubMed_search_articles(query="BRCA2 c.5946delT segregation family", limit=5)`

| Criterion | Condition |
|-----------|-----------|
| **PS3** (Strong path) | Well-established functional assay shows damaging effect. Downgrade to PS3_Supporting for less rigorous assays |
| **BS3** (Strong benign) | Well-established assay shows no functional impact |
| **PP1** (Supporting path) | Co-segregation in multiple affected family members. Upgrade to PP1_Strong at >= 7 meioses |
| **PP4** (Supporting path) | Phenotype highly specific for the gene's disease |

**Nuances**: PubMed returns candidate papers; summarize findings from titles/abstracts. Not all functional assays qualify -- ClinGen gene-specific guidance defines valid assays.

### Criteria Requiring Clinical Data (Not Automated)

PS2 (de novo), PS4 (case-control prevalence), PM3 (in trans with pathogenic), PM6 (assumed de novo), BS4 (no segregation), BP2 (in trans/cis with pathogenic), BP5 (alternate explanation) -- document as "Not Assessed" unless user provides clinical context.

PM4 (protein length change in non-repeat region) and BP3 (in-frame indel in repeat) can be partially assessed from variant type. BP7 (synonymous, no splice impact) assessable via SpliceAI.

---

## Classification Algorithm

Combine criteria at their **applied strength** (after upgrades/downgrades):

### Pathogenic
1. PVS1 + >= 1 Strong | 2. PVS1 + >= 2 Moderate | 3. PVS1 + 1 Moderate + 1 Supporting
4. PVS1 + >= 2 Supporting | 5. >= 2 Strong | 6. 1 Strong + >= 3 Moderate
7. 1 Strong + 2 Moderate + >= 2 Supporting | 8. 1 Strong + 1 Moderate + >= 4 Supporting

### Likely Pathogenic
1. PVS1 + 1 Moderate | 2. 1 Strong + 1-2 Moderate | 3. 1 Strong + >= 2 Supporting
4. >= 3 Moderate | 5. 2 Moderate + >= 2 Supporting | 6. 1 Moderate + >= 4 Supporting

### Benign
1. BA1 (stand-alone) | 2. >= 2 Strong benign

### Likely Benign
1. 1 Strong benign + 1 Supporting benign | 2. >= 2 Supporting benign

### VUS
Criteria do not meet any threshold above, OR pathogenic and benign evidence conflict.

---

## Output Format

```markdown
# ACMG Variant Classification Report

## Variant: [HGVS]
- **Gene**: [symbol] | **Transcript**: [MANE Select] | **Protein**: [p.notation] | **Type**: [variant type]

## Classification: [PATHOGENIC / LIKELY PATHOGENIC / VUS / LIKELY BENIGN / BENIGN]

## Evidence Summary
### Pathogenic Criteria Met
| Criterion | Strength | Evidence | Source |

### Benign Criteria Met
| Criterion | Strength | Evidence | Source |

### Criteria Not Met (key ones with reasoning)
### Criteria Not Assessed (and why)

## Detailed Evidence
- Population: gnomAD AF, ancestry max, homozygotes, gene constraints
- Computational: predictor concordance table
- Clinical: ClinVar classification + review status, CIViC entries
- Domain: InterPro domains, UniProt annotations
- Splice: SpliceAI scores, canonical site status
- Literature: key functional study findings

## Classification Logic
Applied rule: [e.g., "PVS1 + PM2_Supporting = Likely Pathogenic (LP rule 1)"]

## Limitations
- [Criteria not assessed and what data would be needed]
```

---

## Quick Reference: Criteria to Tools

| Criterion | Primary Tool | Fallback |
|-----------|-------------|----------|
| **PVS1** | `EnsemblVEP_annotate_hgvs` + `gnomad_get_gene_constraints` | `VariantValidator_validate_variant` |
| **PS1** | `ClinVar_search_variants` + `civic_get_variants_by_gene` | `MyVariant_query_variants` |
| **PS3** | `PubMed_search_articles` | Manual review |
| **PM1** | `InterPro_get_entries_for_protein` + `UniProt_get_function_by_accession` | `alphafold_get_prediction` |
| **PM2** | `gnomad_get_variant` | `MyVariant_query_variants` |
| **PM5** | `ClinVar_search_variants` + `civic_get_variants_by_gene` | Same-residue search |
| **PP2** | `gnomad_get_gene_constraints` | Literature |
| **PP3** | `MyVariant_query_variants` (REVEL/CADD/SIFT/PolyPhen) | `EnsemblVEP_annotate_hgvs` |
| **PP5** | `ClinVar_search_variants` | `ClinVar_get_variant_details` |
| **BA1** | `gnomad_get_variant` (AF >= 5%) | `MyVariant_query_variants` |
| **BS1** | `gnomad_get_variant` | `MyVariant_query_variants` |
| **BS3** | `PubMed_search_articles` | Manual review |
| **BP1** | Gene mechanism + `gnomad_get_gene_constraints` | Literature |
| **BP4** | `MyVariant_query_variants` (all benign) | `EnsemblVEP_annotate_hgvs` |
| **BP6** | `ClinVar_search_variants` | `ClinVar_get_variant_details` |
| **BP7** | SpliceAI (< 0.1) + synonymous | `EnsemblVEP_annotate_hgvs` |

---

## Common Patterns

**Pattern 1: Known pathogenic frameshift** -- "Classify BRCA2 c.5946delT"
Phase 0 (validate) -> Phase 1 (gnomAD absent, PM2_Supporting) -> Phase 3 (ClinVar Pathogenic, PP5) -> Phase 4 (DNA repair domain, PM1) -> Phase 5 (frameshift + LOF gene, PVS1) -> Phase 6 (literature PS3)
Result: **Pathogenic** (PVS1 + PS3 + PM1 + PM2_Supporting + PP5)

**Pattern 2: Missense VUS** -- "Is BRCA1 p.Arg1699Gln pathogenic?"
Phase 0 -> Phase 1 (rare, PM2_Supporting) -> Phase 2 (REVEL 0.82, CADD 26, PP3) -> Phase 3 (ClinVar VUS) -> Phase 4 (BRCT domain, PM1) -> Phase 6 (reduced activity, PS3_Moderate)
Result: **Likely Pathogenic** (PS3_Moderate + PM1 + PM2_Supporting + PP3)

**Pattern 3: Common benign variant** -- "ACMG for rs1800497"
Phase 1 (gnomAD AF=0.21, BA1) -> short-circuit. Result: **Benign** (BA1 stand-alone)

**Pattern 4: Deep-intronic variant** -- "Classify NM_000059.4:c.7977+100A>G"
Phase 1 (check AF) -> Phase 5 (SpliceAI < 0.1) -> Result: **Likely Benign** or VUS depending on frequency
