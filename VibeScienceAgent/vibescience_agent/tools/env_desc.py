# Copyright 2026 Huawei Technologies Co., Ltd
# Copyright 2025 Biomni
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

# Software libraries and CLI tools available in the environment

library_content_dict = {
    # Core Bioinformatics Libraries
    "biopython": "[Python Package] A set of tools for biological computation including parsers for bioinformatics files, access to online services, and interfaces to common bioinformatics programs.",
    "biom-format": "[Python Package] The Biological Observation Matrix (BIOM) format is designed for representing biological sample by observation contingency tables with associated metadata.",
    "scanpy": "[Python Package] A scalable toolkit for analyzing single-cell gene expression data, specifically designed for large datasets using AnnData.",
    "scikit-bio": "[Python Package] Data structures, algorithms, and educational resources for bioinformatics, including sequence analysis, phylogenetics, and ordination methods.",
    "anndata": "[Python Package] A Python package for handling annotated data matrices in memory and on disk, primarily used for single-cell genomics data.",
    "mudata": "[Python Package] A Python package for multimodal data storage and manipulation, extending AnnData to handle multiple modalities.",
    "pyliftover": "[Python Package] A Python implementation of UCSC liftOver tool for converting genomic coordinates between genome assemblies.",
    "biopandas": "[Python Package] A package that provides pandas DataFrames for working with molecular structures and biological data.",
    "biotite": "[Python Package] A comprehensive library for computational molecular biology, providing tools for sequence analysis, structure analysis, and more.",
    "lazyslide": "[Python Package] A Python framework that brings interoperable, reproducible whole slide image analysis, enabling seamless histopathology workflows from preprocessing to deep learning.",
    # Genomics & Variant Analysis
    "gget": "[Python Package] A toolkit for accessing genomic databases and retrieving sequences, annotations, and other genomic data.",
    "lifelines": "[Python Package] A complete survival analysis library for fitting models, plotting, and statistical tests.",
    "gseapy": "[Python Package] A Python wrapper for Gene Set Enrichment Analysis (GSEA) and visualization.",
    "scrublet": "[Python Package] A tool for detecting doublets in single-cell RNA-seq data.",
    "cellxgene-census": "[Python Package] A tool for accessing and analyzing the CellxGene Census, a collection of single-cell datasets. To download a dataset, use the download_source_h5ad function with the dataset id as the argument (856c1b98-5727-49da-bf0f-151bdb8cb056, no .h5ad extension).",
    "hyperopt": "[Python Package] A Python library for optimizing hyperparameters of machine learning algorithms.",
    "scvelo": "[Python Package] A tool for RNA velocity analysis in single cells using dynamical models.",
    "pysam": "[Python Package] A Python module for reading, manipulating and writing genomic data sets in SAM/BAM/VCF/BCF formats.",
    "pyfaidx": "[Python Package] A Python package for efficient random access to FASTA files.",
    "pyranges": "[Python Package] A Python package for interval manipulation with a pandas-like interface.",
    "pybedtools": "[Python Package] A Python wrapper for Aaron Quinlan's BEDTools programs.",
    # Structural Biology & Drug Discovery
    "rdkit": "[Python Package] A collection of cheminformatics and machine learning tools for working with chemical structures and drug discovery.",
    "deeppurpose": "[Python Package] A deep learning library for drug-target interaction prediction and virtual screening.",
    "pyscreener": "[Python Package] A Python package for virtual screening of chemical compounds.",
    "openbabel": "[Python Package] A chemical toolbox designed to speak the many languages of chemical data, supporting file format conversion and molecular modeling.",
    "descriptastorus": "[Python Package] A library for computing molecular descriptors for machine learning applications in drug discovery.",
    "openmm": "[Python Package] A toolkit for molecular simulation using high-performance GPU computing.",
    "pytdc": "[Python Package] A Python package for Therapeutics Data Commons, providing access to machine learning datasets for drug discovery.",
    # Data Science & Statistical Analysis
    "pandas": "[Python Package] A fast, powerful, and flexible data analysis and manipulation library for Python.",
    "numpy": "[Python Package] The fundamental package for scientific computing with Python, providing support for arrays, matrices, and mathematical functions.",
    "scipy": "[Python Package] A Python library for scientific and technical computing, including modules for optimization, linear algebra, integration, and statistics.",
    "scikit-learn": "[Python Package] A machine learning library featuring various classification, regression, and clustering algorithms.",
    "matplotlib": "[Python Package] A comprehensive library for creating static, animated, and interactive visualizations in Python.",
    "seaborn": "[Python Package] A statistical data visualization library based on matplotlib with a high-level interface for drawing attractive statistical graphics.",
    "statsmodels": "[Python Package] A Python module for statistical modeling and econometrics, including descriptive statistics and estimation of statistical models.",
    "pymc3": "[Python Package] A Python package for Bayesian statistical modeling and probabilistic machine learning.",
    "umap-learn": "[Python Package] Uniform Manifold Approximation and Projection, a dimension reduction technique.",
    "faiss-cpu": "[Python Package] A library for efficient similarity search and clustering of dense vectors.",
    "harmony-pytorch": "[Python Package] A PyTorch implementation of the Harmony algorithm for integrating single-cell data.",
    # General Bioinformatics & Computational Utilities
    "tiledb": "[Python Package] A powerful engine for storing and analyzing large-scale genomic data.",
    "tiledbsoma": "[Python Package] A library for working with the SOMA (Stack of Matrices) format using TileDB.",
    "h5py": "[Python Package] A Python interface to the HDF5 binary data format, allowing storage of large amounts of numerical data.",
    "tqdm": "[Python Package] A fast, extensible progress bar for loops and CLI applications.",
    "joblib": "[Python Package] A set of tools to provide lightweight pipelining in Python, including transparent disk-caching and parallel computing.",
    "opencv-python": "[Python Package] OpenCV library for computer vision tasks, useful for image analysis in biological contexts.",
    "PyPDF2": "[Python Package] A library for working with PDF files, useful for extracting text from scientific papers.",
    "googlesearch-python": "[Python Package] A library for performing Google searches programmatically.",
    "scikit-image": "[Python Package] A collection of algorithms for image processing in Python.",
    "pymed": "[Python Package] A Python library for accessing PubMed articles.",
    "arxiv": "[Python Package] A Python wrapper for the arXiv API, allowing access to scientific papers.",
    "scholarly": "[Python Package] A module to retrieve author and publication information from Google Scholar.",
    "cryosparc-tools": "[Python Package] Tools for working with cryoSPARC, a platform for cryo-EM data processing.",
    "mageck": "[Python Package] Analysis of CRISPR screen data.",
    "igraph": "[Python Package] Network analysis and visualization.",
    "pyscenic": "[Python Package] Analysis of single-cell RNA-seq data and gene regulatory networks.",
    "cooler": "[Python Package] Storage and analysis of Hi-C data.",
    "trackpy": "[Python Package] Particle tracking in images and video.",
    "nnunet": "[Python Package] A deep learning framework for biomedical image segmentation, providing a standardized approach to training and inference.",
    "cellpose": "[Python Package] Cell segmentation in microscopy images.",
    "viennarna": "[Python Package] RNA secondary structure prediction.",
    "PyMassSpec": "[Python Package] Mass spectrometry data analysis.",
    "python-libsbml": "[Python Package] Working with SBML files for computational biology.",
    "cobra": "[Python Package] Constraint-based modeling of metabolic networks.",
    "reportlab": "[Python Package] Creation of PDF documents.",
    "flowkit": "[Python Package] Toolkit for processing flow cytometry data.",
    "hmmlearn": "[Python Package] Hidden Markov model analysis.",
    "msprime": "[Python Package] Simulation of genetic variation.",
    "tskit": "[Python Package] Handling tree sequences and population genetics data.",
    "cyvcf2": "[Python Package] Fast parsing of VCF files.",
    "pykalman": "[Python Package] Kalman filter and smoother implementation.",
    "fanc": "[Python Package] Analysis of chromatin conformation data.",
    "loompy": "A Python implementation of the Loom file format for efficiently storing and working with large omics datasets.",
    "pyBigWig": "A Python library for accessing bigWig and bigBed files for genome browser track data.",
    "pymzml": "A Python module for high-throughput bioinformatics analysis of mass spectrometry data.",
    "optlang": "A Python package for modeling optimization problems symbolically.",
    "FlowIO": "A Python package for reading and writing flow cytometry data files.",
    "FlowUtils": "Utilities for processing and analyzing flow cytometry data.",
    "arboreto": "A Python package for inferring gene regulatory networks from single-cell RNA-seq data.",
    "pdbfixer": "A Python package for fixing problems in PDB files in preparation for molecular simulations.",
}
