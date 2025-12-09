"""
Matformer Data Module
---------------------

This subpackage provides data loading, preprocessing, and feature construction
for the **Matformer** model. It is independent of MindScience's core `data` module
to allow customized graph-based molecular representations.

Main Components:
    - data.py: Core dataset management and property mapping.
    - features.py: Feature engineering for atomic and bond attributes.
    - generate.py: Functions to build and prepare training datasets.
    - graphs.py: Molecular graph construction and neighborhood computation.

Typical Usage:
    from data.generate import get_prop_model

    dataset = get_prop_model(
        dataset_path="datasets/mptrj_ase.db",
        task="property_prediction"
    )
"""
