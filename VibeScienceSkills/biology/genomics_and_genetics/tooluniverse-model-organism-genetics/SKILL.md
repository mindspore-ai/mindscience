---
name: tooluniverse-model-organism-genetics
description: Cross-species genetic analysis using model organism databases. Maps human genes to orthologs in mouse, fly, worm, zebrafish, yeast, and frog, then retrieves phenotypes, expression, and functional data from MGI, FlyBase, WormBase, ZFIN, SGD, and Xenbase. Use when users ask about model organisms, gene orthologs, mouse phenotypes, fly genetics, worm RNAi, zebrafish morphants, cross-species comparison, animal models for human disease, or conservation of gene function.
license: MIT License
metadata:
    skill-author: ToolUniverse (https://github.com/mims-harvard/TxAgent)
---

# Model Organism Genetics Pipeline

Map human genes to model organism orthologs and retrieve phenotype, expression, and functional data across six species. Synthesize cross-species evidence to assess gene function conservation and identify the best animal models for studying human genes and diseases.

## Key Principles

1. **Disambiguation FIRST** - Resolve human gene to canonical identifiers before ortholog mapping
2. **Ortholog confidence matters** - Prioritize one-to-one orthologs; flag one-to-many or many-to-many
3. **Species-appropriate IDs** - Each database uses its own ID format (MGI:xxx, FB:FBgnXXX, WBGene, ZFIN:ZDB-GENE, SGD:SXXX)
4. **Negative results are data** - "No ortholog in yeast" is informative (gene may be metazoan-specific)
5. **Evidence grading** - T1: direct experimental (knockout/KD phenotype), T2: genetic screen, T3: computational orthology, T4: sequence similarity only
6. **Organism selection by question** - Not all organisms are relevant to all questions (see Organism Selection Guide)
7. **Synthesize, don't enumerate** - The value is in connecting phenotypes across species, not listing them separately. Always end with a cross-species synthesis that answers "what does this gene do?" at a conceptual level
8. **Report progressively** - Create report file first, populate section by section
9. **English-first queries** - Use English gene symbols in all tool calls; respond in user's language

## When to Use

Apply when users ask about:
- "What is the mouse ortholog of [human gene]?"
- "What happens when you knock out [gene] in mouse/fly/worm/fish?"
- "Which model organism best models [human disease]?"
- Cross-species gene function comparison
- Conservation of a gene across evolution
- Animal models for a specific human gene or pathway
- Phenotypes from genetic screens in model organisms

**Not for** (use other skills): human variant interpretation (`tooluniverse-variant-analysis`), drug target validation (`tooluniverse-drug-target-validation`), human disease characterization (`tooluniverse-multiomic-disease-characterization`).

## Input Parameters

| Parameter | Required | Description | Example |
|-----------|----------|-------------|---------|
| **gene** | Yes | Human gene symbol, Ensembl ID, or UniProt ID | `TP53`, `ENSG00000141510` |
| **organisms** | No | Subset of organisms to query (default: all) | `["mouse", "fly", "worm"]` |
| **disease** | No | Human disease context for model relevance | `Parkinson's disease` |
| **focus** | No | Analysis focus: `phenotypes`, `expression`, `function`, `all` | `phenotypes` |

---

## Pipeline Phases

### Phase 0: Human Gene Disambiguation (ALWAYS FIRST)

Resolve input to all needed identifiers.

**Steps**:
1. `MyGene_query_genes(query="<gene>")` - Get Ensembl ID, Entrez ID, UniProt, symbol
   - IMPORTANT: Filter results by `symbol` match (first hit may be a LOC pseudogene)
2. `ensembl_lookup_gene(gene_id="<ensembl_id>", species="homo_sapiens")` - Validate Ensembl ID
3. If disease context provided: `HPO_search_terms(query="<disease>")` - Get HPO terms for phenotype matching

**Output**: Canonical symbol, Ensembl ID (ENSG), Entrez ID, UniProt accession, full gene name.

**Failure handling**: If gene not found, try `UniProt_search(query="<gene>", organism="9606")` as fallback.

---

### Phase 1: Ortholog Mapping

Map human gene to orthologs across all target species.

**Primary tool**: `EnsemblCompara_get_orthologues`
```
EnsemblCompara_get_orthologues(
    gene="<ensembl_id>",        # e.g., "ENSG00000141510"
    species="human",
    target_species="<species>"  # Use EXACT names below
)
```

**CRITICAL: Ensembl Compara species names** (these are the only accepted values):
- Mouse: `"mouse"` or `"mus_musculus"`
- Zebrafish: `"zebrafish"` or `"danio_rerio"`
- Fly: `"drosophila_melanogaster"` (NOT "fruitfly" — that returns HTTP 400)
- Worm: `"caenorhabditis_elegans"`
- Yeast: `"saccharomyces_cerevisiae"`
- Frog: `"xenopus_tropicalis"`

**Fallback tools** (if Ensembl Compara returns no results or errors):
1. `PANTHER_ortholog(gene_id="<gene_symbol>", organism=9606, target_organism=<taxon>)` - Taxon IDs: mouse=10090, fly=7227, worm=6239, zebrafish=7955, yeast=559292, frog=8364
2. `NCBIDatasets_get_orthologs(gene_id="<entrez_id>")` - Broad ortholog search (vertebrates mainly)
3. **For distant orthologs (especially fly)**: If automated tools fail, use `FlyMine_search(query="<human_gene_symbol>")` with text search. Distant orthologs (low overall sequence identity but conserved domains) often fail automated detection. FlyBase gene IDs can be found this way, then confirmed with `FlyBase_get_gene_orthologs`.
4. **For worm**: `WormBase_get_gene(gene_id="<gene_symbol>")` may return ortholog information directly in the gene record.

**Additional per-organism ortholog tools** (confirm/supplement):
- `FlyBase_get_gene_orthologs(gene_id="FB:FBgnXXXXXXX")` - Fly-centric ortholog view
- `ZFIN_get_gene_orthologs(gene_id="ZFIN:ZDB-GENE-XXXXXX-X")` - Zebrafish-centric view

**Cross-reference with Monarch**:
- `Monarch_search_gene(query="<gene_symbol>")` - Get Monarch gene entity
- `MonarchV3_get_associations(subject="HGNC:<id>", category="biolink:GeneHomologAssociation")` - All orthologs via Monarch

**Finding organism-specific gene IDs when ortholog mapping fails**:
When automated ortholog mapping misses a hit (common for distant orthologs like fly or worm):
1. Search the organism database directly by human gene symbol (e.g., `FlyBase_get_gene` with the expected fly gene name)
2. Fly gene names often follow the pattern: human FOXP2 → fly FoxP (capitalization differs)
3. If the gene family is known (e.g., forkhead), search by family name
4. Use `FlyMine_search` or `WormBase_get_gene` for text-based discovery

**Output per ortholog**: Species, gene symbol, gene ID, homology type (1:1, 1:many, many:many), percent identity, confidence.

**Key interpretation**:
- **One-to-one**: Strong conservation; model organism gene likely has conserved function
- **One-to-many**: Gene duplicated in target species; check subfunctionalization
- **No ortholog found by tools**: Try manual search (step 3-4 above) before concluding absence. Sequence divergence ≠ functional divergence — a gene can have a conserved role despite low sequence identity
- **True absence**: Gene may be lineage-specific (e.g., adaptive immunity genes absent in invertebrates)

**Paralog contamination warning**: Gene families (e.g., FOXP1/FOXP2/FOXP3/FOXP4, or HOX clusters) generate false ortholog hits. To distinguish true orthologs from paralogs:
- Check synteny (conserved gene order in the genome neighborhood)
- Verify the ortholog mapping is 1:1 (paralogs show 1:many or many:many)
- If the target species has a single gene in the family (e.g., Drosophila has one FoxP vs four human FOXPs), it is the co-ortholog of all human paralogs — note this explicitly

---

### Phase 2: Mouse Phenotypes (MGI)

Mouse is the primary mammalian model. Knockout phenotypes are the gold standard for gene function.

**Steps**:
1. `MGI_search_genes(query="<mouse_symbol>")` - Confirm mouse gene ID (MGI:XXXXXXX)
2. `MGI_get_gene(gene_id="MGI:XXXXXXX")` - Full gene details, synonyms, location
3. `MGI_get_phenotypes(gene_id="MGI:XXXXXXX", limit=50)` - Knockout/transgenic phenotypes

**What to extract**:
- Phenotype terms (MP ontology) and their descriptions
- Allele types: knockout (null), conditional KO, point mutation, transgenic
- Zygosity: homozygous vs heterozygous effects
- Lethality: embryonic lethal, perinatal lethal, viable
- Disease model relevance: does phenotype match human disease?

**If disease context provided**: Match mouse phenotypes to HPO terms from Phase 0.

**Phenotype-to-disease mapping via Monarch**:
- `MonarchV3_get_associations(subject="MGI:XXXXXXX", category="biolink:GeneToPhenotypicFeatureAssociation")` - Phenotype associations
- `MonarchV3_get_associations(subject="MGI:XXXXXXX", category="biolink:GeneToDiseaseAssociation")` - Disease model annotations

---

### Phase 3: Invertebrate Models (Fly and Worm)

Drosophila and C. elegans offer powerful genetic screens, rapid generation time, and conserved pathways.

#### 3A: Drosophila (FlyBase)

1. `FlyBase_get_gene(gene_id="FB:FBgnXXXXXXX")` - Gene details, function summary
2. `FlyBase_get_gene_alleles(gene_id="FB:FBgnXXXXXXX", limit=20)` - Mutant alleles (LOF, GOF, RNAi lines)
3. `FlyBase_get_gene_disease_models(gene_id="FB:FBgnXXXXXXX")` - Human disease models in fly
4. `FlyBase_get_gene_expression(gene_id="FB:FBgnXXXXXXX")` - Tissue/stage expression
5. `FlyBase_get_gene_interactions(gene_id="FB:FBgnXXXXXXX")` - Genetic and physical interactions

**What to extract**:
- Loss-of-function phenotypes (lethal, viable, morphological, behavioral)
- Gain-of-function phenotypes (overexpression, misexpression)
- Genetic interaction partners (enhancers, suppressors)
- Human disease model annotations (OMIM cross-references)
- Expression pattern (ubiquitous vs tissue-specific)

#### 3B: C. elegans (WormBase)

1. `WormBase_get_gene(gene_id="WBGene00XXXXXX")` - Gene details, concise description
2. `WormBase_get_phenotypes(gene_id="WBGene00XXXXXX")` - RNAi and mutant phenotypes
3. `WormBase_get_expression(gene_id="WBGene00XXXXXX")` - Expression pattern

**What to extract**:
- RNAi phenotypes (genome-wide screen results)
- Mutant phenotypes (null, reduction-of-function, gain-of-function)
- Essential vs non-essential classification
- Tissue expression (neurons, muscle, intestine, germline)

---

### Phase 4: Vertebrate Models (Zebrafish and Frog)

Zebrafish and Xenopus are key for developmental biology, organ formation, and disease modeling.

#### 4A: Zebrafish (ZFIN)

1. `ZFIN_get_gene(gene_id="ZFIN:ZDB-GENE-XXXXXX-X")` - Gene details
2. `ZFIN_get_gene_phenotypes(gene_id="ZFIN:ZDB-GENE-XXXXXX-X", limit=30)` - Morpholino/CRISPR/mutant phenotypes
3. `ZFIN_get_gene_alleles(gene_id="ZFIN:ZDB-GENE-XXXXXX-X", limit=20)` - Available mutant lines
4. `ZFIN_get_gene_expression(gene_id="ZFIN:ZDB-GENE-XXXXXX-X")` - Spatiotemporal expression

**What to extract**:
- Morpholino knockdown phenotypes (rapid, but potential off-target)
- CRISPR mutant phenotypes (more reliable)
- ENU mutant phenotypes (unbiased forward genetics)
- Expression domains during development (in situ hybridization data)
- Maternal vs zygotic contribution

#### 4B: Frog (Xenbase)

1. `Xenbase_search_genes(query="<gene_symbol>")` - Find Xenopus gene
2. `Xenbase_get_gene(gene_id="<xenbase_id>")` - Gene details, expression, phenotypes

**What to extract**:
- Morpholino/CRISPR phenotypes
- Expression in developmental stages (Nieuwkoop-Faber stages)
- Note: X. laevis is allotetraploid (two homeologs: .L and .S)

---

### Phase 5: Yeast Functional Genomics (SGD)

Saccharomyces cerevisiae provides fundamental cell biology insights for conserved cellular processes.

1. `SGD_search(query="<gene_symbol>", category="gene")` - Find yeast gene (SGD ID)
2. `SGD_get_gene(sgd_id="<sgd_id>")` - Gene details, function, pathway
3. `SGD_get_phenotypes(sgd_id="<sgd_id>")` - Deletion and overexpression phenotypes
4. `SGD_get_go_annotations(sgd_id="<sgd_id>")` - GO terms (molecular function, biological process, cellular component)
5. `SGD_get_interactions(sgd_id="<sgd_id>")` - Genetic and physical interactions

**What to extract**:
- Deletion viability (essential vs non-essential)
- Growth phenotypes under stress conditions (DNA damage, oxidative stress, nutrient deprivation)
- Synthetic lethal interactions (potential drug targets)
- GO annotations (often the best-characterized for conserved genes)
- Physical interaction partners (protein complexes)

**When yeast is most informative**: Cell cycle, DNA repair, protein folding, metabolism, autophagy, secretory pathway, chromatin biology.

**When yeast is NOT informative**: Multicellular processes (development, immunity, neural function).

---

### Phase 6: Cross-Species Pathway Conservation

Assess whether the gene's pathway context is conserved.

**Steps**:
1. Collect orthologs from all species (Phase 1 results)
2. `STRING_get_network(identifiers="<human_gene> <mouse_ortholog> <fly_ortholog>", species=9606)` - Check if interaction partners are also conserved
3. `ReactomeAnalysis_pathway_enrichment(identifiers="<human_gene> <ortholog1> <ortholog2>")` - Shared pathway membership

**Cross-species phenotype comparison via Monarch**:
- `MonarchV3_get_associations(subject="HGNC:<id>", category="biolink:GeneToPhenotypicFeatureAssociation")` - Human gene phenotypes
- Compare with mouse (MP), fly (FBcv), worm (WBPhenotype) phenotypes from Phases 2-3
- `MonarchV3_phenotype_similarity_search(...)` - Automated phenotype comparison across species

**Conservation assessment criteria**:
- **Highly conserved**: Ortholog in all 6 species, consistent phenotypes, shared pathways
- **Vertebrate-specific**: Ortholog in mouse/fish/frog but not fly/worm/yeast
- **Metazoan-specific**: Ortholog in mouse/fish/fly/worm but not yeast
- **Mammalian-specific**: Ortholog only in mouse (and frog sometimes)
- **Human-specific**: No clear ortholog in any model organism

---

## Phase 7: Cross-Species Phenotype Synthesis (CRITICAL)

This phase is what makes the skill useful — it transforms per-organism data into biological insight.

**Objective**: Connect phenotypes across species to identify the gene's core function and evolutionary trajectory.

### Step 1: Build a Cross-Species Phenotype Matrix

| Feature | Human | Mouse | Fly | Worm | Zebrafish | Yeast |
|---------|-------|-------|-----|------|-----------|-------|
| Ortholog present? | -- | | | | | |
| Lethality (LOF) | | | | | | |
| Primary phenotype | | | | | | |
| Expression domain | | | | | | |
| Conserved pathway | | | | | | |

### Step 2: Identify the Ancestral/Core Function

Look for the phenotype that is most consistent across species. This is often more abstract than any single organism's phenotype:
- Mouse "reduced vocalization" + Fly "defective courtship song" + Human "speech apraxia" → **Core function: motor circuit development and learned motor sequences**
- Mouse "embryonic lethal" + Worm "lethal" + Yeast "essential" → **Core function: fundamental cell viability**
- Mouse "cardiac defects" + Zebrafish "heart edema" + Human "cardiomyopathy" → **Core function: cardiac development**

### Step 3: Map Phenotype Ontology Across Species

Different species use different phenotype ontologies. To compare meaningfully:

| Species | Ontology | Tool for Mapping |
|---------|----------|-----------------|
| Human | HPO (Human Phenotype Ontology) | `HPO_search_terms` |
| Mouse | MP (Mammalian Phenotype) | `MGI_get_phenotypes` |
| Fly | FBcv (FlyBase Controlled Vocabulary) | `FlyBase_get_gene` (phenotype descriptions) |
| Worm | WBPhenotype | `WormBase_get_phenotypes` |
| Zebrafish | ZP (Zebrafish Phenotype) | `ZFIN_get_gene_phenotypes` |

**Cross-species phenotype matching**: Use `MonarchV3_phenotype_similarity_search` to find equivalent phenotypes across species via the uPheno ontology (which maps MP↔HPO↔ZP↔WBPhenotype↔FBcv).

When automated mapping fails, use biological reasoning:
- "Reduced ultrasonic vocalizations" (mouse) ≈ "Defective courtship song" (fly) ≈ "Speech apraxia" (human) — all are **disrupted learned motor sequences for communication**
- "Embryonic lethal" (mouse) ≈ "Lethal" (worm) ≈ "Inviable" (yeast) — all are **essential for viability**
- "Purkinje cell defects" (mouse) ≈ "Protocerebral bridge defects" (fly) — both are **motor coordination center malformation**

### Step 4: Synthesize the Evolutionary Narrative

Write a synthesis paragraph that answers:
1. **What is this gene's fundamental role?** (Abstract from species-specific phenotypes)
2. **How has the function been elaborated during evolution?** (e.g., motor circuit → vocal circuit → speech)
3. **Which model organism best recapitulates the human condition?** (Consider phenotype match AND experimental tractability)
4. **What are the key conserved vs divergent aspects?** (e.g., conserved neural expression but divergent behavioral output)

### Step 5: Recommend the Best Model System

Based on the synthesis, recommend which organism(s) to use for further study, considering:
- **Phenotype match**: Which organism's phenotype most closely mirrors the human condition?
- **Experimental tools**: Which organism has the best genetic tools for the question?
- **Complementary models**: Often the best approach uses 2-3 organisms (e.g., mouse for physiology + fly for genetic screens)
- **Practical considerations**: Cost, time, throughput, imaging capability

---

## Phase 8: Conserved Regulatory Elements (Optional)

For developmental biology questions, conserved regulatory elements (enhancers, promoters) are often as important as the coding sequence.

**Tools**:
- `EnsemblReg_get_regulatory_elements(species="homo_sapiens", region="<chr:start-end>")` — find regulatory features near the gene
- `ENCODE_search_experiments(query="<gene_symbol> ChIP-seq")` — ChIP-seq and ATAC-seq data
- `UCSC_get_encode_cCREs(genome="hg38", chrom="<chr>", start=<start>, end=<end>)` — candidate cis-regulatory elements
- `JASPAR_search_motifs(query="<TF_name>")` — transcription factor binding motifs
- `UCSC_get_track(genome="hg38", track="phastCons100way", chrom="<chr>", start=<start>, end=<end>)` — conservation scores for non-coding regions

**Workflow**:
1. Get the genomic coordinates of the human gene (from Phase 0)
2. Extend the region (±100kb for nearby enhancers)
3. Query ENCODE for regulatory elements and UCSC for conservation
4. Look for deeply conserved non-coding elements (CNEs) — these often control developmental expression
5. Cross-reference with expression patterns from Phases 2-4

**When to use**: Always for developmental biology questions. Deeply conserved enhancers (e.g., FOXP2 intron 8 enhancer) can be more functionally important than the protein sequence itself.

---

## Phase 9: Human Disease Connection (Optional)

Link model organism findings back to human disease.

**Tools**:
- `OMIM_search(query="<gene_symbol>")` — Mendelian disease associations
- `ClinVar_search_variants(query="<gene_symbol>")` — pathogenic variants
- `ClinGen_search_gene_validity(gene="<gene_symbol>")` — gene-disease validity assessment
- `HPO_search_terms(query="<disease_name>")` — human phenotype terms for cross-species comparison

**Workflow**:
1. Search OMIM for known disease associations
2. Get ClinGen validity level (Definitive, Strong, Moderate, Limited)
3. Search HPO for relevant phenotype terms
4. Map HPO terms back to model organism phenotypes (Phase 7) to assess model fidelity

---

## Organism Selection Guide

| Question Type | Best Organism(s) | Rationale |
|---------------|-------------------|-----------|
| Disease model (cancer, neurodegeneration) | Mouse | Mammalian physiology, drug testing |
| Developmental biology | Zebrafish, Frog | Transparent embryos, rapid development |
| Neural circuits / behavior | Fly, Worm | Mapped connectome, genetic tools |
| Cell cycle / DNA repair | Yeast | Best-characterized, rapid genetics |
| Genetic screens (large-scale) | Fly, Worm | RNAi libraries, balancer chromosomes |
| Protein trafficking / secretion | Yeast | Sec pathway first characterized here |
| Immune system / inflammation | Mouse | Adaptive immunity (absent in invertebrates) |
| Cardiac development | Zebrafish | Regeneration, transparent, genetic tools |
| Aging / lifespan | Worm, Fly, Yeast | Short lifespan, conserved pathways |
| Metabolism / obesity | Mouse, Fly | Diet-induced obesity models |
| Signaling pathways (Notch, Wnt, Hh) | Fly | Originally discovered in Drosophila |
| Chromatin / epigenetics | Yeast, Fly | Histone code, Polycomb/Trithorax |

---

## Quick Reference: Tool-to-Organism Mapping

### Ortholog Mapping Tools

| Tool | Parameter | Notes |
|------|-----------|-------|
| `EnsemblCompara_get_orthologues` | `gene` (Ensembl ID), `species`, `target_species` | Primary. Use species names: "mouse", "zebrafish", "drosophila_melanogaster" (NOT "fruitfly"), "caenorhabditis_elegans", "saccharomyces_cerevisiae", "xenopus_tropicalis" |
| `PANTHER_ortholog` | `gene_id` (symbol), `organism` (taxon), `target_organism` (taxon) | Fallback. Taxon: 9606/10090/7227/6239/7955/559292/8364 |
| `NCBIDatasets_get_orthologs` | `gene_id` (Entrez ID) | Broad, all species at once |

### Mouse (MGI)

| Tool | Parameter | Returns |
|------|-----------|---------|
| `MGI_search_genes` | `query` (symbol/name) | Gene list with MGI IDs |
| `MGI_get_gene` | `gene_id` (MGI:XXXXXXX) | Full gene record |
| `MGI_get_phenotypes` | `gene_id`, `limit` | MP phenotype annotations |

### Fly (FlyBase)

| Tool | Parameter | Returns |
|------|-----------|---------|
| `FlyBase_get_gene` | `gene_id` (FB:FBgnXXX) | Gene record, function |
| `FlyBase_get_gene_alleles` | `gene_id`, `limit` | Mutant alleles |
| `FlyBase_get_gene_disease_models` | `gene_id` | Human disease models |
| `FlyBase_get_gene_expression` | `gene_id` | Expression pattern |
| `FlyBase_get_gene_interactions` | `gene_id` | Genetic + physical |
| `FlyBase_get_gene_orthologs` | `gene_id`, `stringency` | Ortholog list |

### Worm (WormBase)

| Tool | Parameter | Returns |
|------|-----------|---------|
| `WormBase_get_gene` | `gene_id` (WBGeneXXX) | Gene record |
| `WormBase_get_phenotypes` | `gene_id` | RNAi + mutant phenotypes |
| `WormBase_get_expression` | `gene_id` | Expression pattern |

### Zebrafish (ZFIN)

| Tool | Parameter | Returns |
|------|-----------|---------|
| `ZFIN_get_gene` | `gene_id` (ZFIN:ZDB-GENE-XXX) | Gene record |
| `ZFIN_get_gene_phenotypes` | `gene_id`, `limit` | Morpholino/CRISPR phenotypes |
| `ZFIN_get_gene_alleles` | `gene_id`, `limit` | Mutant lines |
| `ZFIN_get_gene_expression` | `gene_id` | In situ expression |
| `ZFIN_get_gene_orthologs` | `gene_id`, `stringency` | Ortholog list |

### Yeast (SGD)

| Tool | Parameter | Returns |
|------|-----------|---------|
| `SGD_search` | `query`, `category` ("gene") | Gene list with SGD IDs |
| `SGD_get_gene` | `sgd_id` | Gene record, function |
| `SGD_get_phenotypes` | `sgd_id` | Deletion/overexpression phenotypes |
| `SGD_get_go_annotations` | `sgd_id` | GO terms |
| `SGD_get_interactions` | `sgd_id` | Genetic + physical interactions |

### Frog (Xenbase)

| Tool | Parameter | Returns |
|------|-----------|---------|
| `Xenbase_search_genes` | `query`, `species`, `limit` | Gene list |
| `Xenbase_get_gene` | `gene_id` | Gene record |

### Cross-Species / Phenotype Ontology

| Tool | Parameter | Returns |
|------|-----------|---------|
| `HPO_search_terms` | `query` | Human Phenotype Ontology terms |
| `Monarch_search_gene` | `query` | Gene entity across species |
| `MonarchV3_get_associations` | `subject` (gene ID), `category` | Phenotype/disease associations |
| `MonarchV3_phenotype_similarity_search` | phenotype IDs | Cross-species phenotype match |
| `STRING_get_network` | `identifiers` (space-separated), `species` (9606) | Protein interaction network |
| `ReactomeAnalysis_pathway_enrichment` | `identifiers` (space-separated) | Shared pathways |

---

## Common Use Patterns

### Pattern 1: "What does gene X do?" (Full cross-species survey)
Run all phases (0-6). Start with disambiguation, map orthologs, then query all organism databases. Synthesize: which phenotypes are conserved? Where does the gene appear essential?

### Pattern 2: "Best model for disease Y involving gene X"
Phase 0 (disambiguate) -> Phase 1 (orthologs) -> Check `FlyBase_get_gene_disease_models` -> `MGI_get_phenotypes` -> `MonarchV3_get_associations` for disease annotations. Recommend organism where phenotype best recapitulates human disease.

### Pattern 3: "Is gene X essential?"
Phase 0 -> Phase 1 -> Check: Mouse KO lethality (Phase 2), Fly LOF lethality (Phase 3A), Worm RNAi lethality (Phase 3B), Yeast deletion viability (Phase 5). Cross-reference: essential in all species = core cellular function.

### Pattern 4: "Conservation of gene X across evolution"
Phase 0 -> Phase 1 (all species) -> Phase 6 (pathway conservation). Report: ortholog presence/absence per species, sequence identity, shared vs divergent phenotypes, pathway membership.

### Pattern 5: "Genetic interactors of gene X"
Phase 0 -> Phase 1 -> `FlyBase_get_gene_interactions` + `SGD_get_interactions` + `STRING_get_network`. Compare: which interactors are conserved across species? Synthetic lethal partners in yeast = potential drug targets.

---

## Completeness Checklist

Before finalizing any report, verify:

- [ ] Human gene resolved to Ensembl ID, Entrez ID, UniProt, symbol
- [ ] Ortholog mapping attempted for all requested species
- [ ] Each ortholog identified with confidence level (1:1, 1:many, none)
- [ ] Phenotype data retrieved for each species with orthologs
- [ ] Expression data included where available
- [ ] "No ortholog" or "No data" explicitly stated (not silently omitted)
- [ ] Cross-species conservation summary provided
- [ ] Organism recommendation given if disease context provided
- [ ] Evidence graded (T1-T4) for key findings
- [ ] All tool calls documented with actual parameters used
