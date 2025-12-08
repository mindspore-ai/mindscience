"""
Matformer Model Package
-----------------------

This package defines the core neural network architectures used in the
Matformer application of MindChem.

Typical contents of this package include:
    - Backbone definitions for the Matformer encoder.
    - Embedding and positional encoding modules.
    - Readout / property head for energy, force or other scalar targets.
    - Utility functions for building models from config dictionaries.

The package is designed to be:
    - Modular: different components (embedding, attention blocks, heads)
      can be swapped or extended.
    - Config-driven: model hyper-parameters are usually specified in
      YAML / JSON config files and parsed into constructor arguments.
    - MindSpore-friendly: all models are implemented with MindSpore
      `nn.Cell`, supporting graph mode and Ascend devices."""
