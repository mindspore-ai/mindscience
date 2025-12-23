MindSpore Science Documentation
================================

Introduction
------------

Combining AI with scientific computing (i.e., AI+Scientific Computing) refers to using artificial intelligence technologies (such as machine learning, deep learning, etc.) to compute and analyze scientific problems. This combination breaks through the limitations of traditional mathematical models and algorithms, leveraging AI's powerful computing capabilities to explore unknown fields and improve computational efficiency and accuracy. The capabilities of AI+Scientific Computing mainly come from big data support and algorithm optimization. Through continuous learning and iteration, AI can better handle complex scientific computing problems.

MindSpore Science is a high-performance scientific computing industry suite built on the MindSpore fusion architecture, providing industry-leading datasets, SOTA models, common model interfaces, and pre/post-processing tools. This suite is deeply optimized for Ascend, accelerating scientific industry application development.

Supported Features
------------------

MindScience-Core is the core component of MindScience, providing developers with easy-to-use, Ascend-friendly common model interfaces to accelerate AI4Science model development. It mainly provides the following capabilities:

Common Model Interfaces:

- Graph Computing: Due to the natural adaptation of graph topology to data forms such as grids and molecular structures, models in the AI4Science field extensively adopt architectures centered on GNN (Graph Neural Network). MindScience provides a comprehensive set of GNN interfaces that are deeply integrated with the Ascend platform while maintaining compatibility with mainstream framework interfaces in the industry, effectively improving the development efficiency of related models.

- Equivariant Computing: The equivariant computing library e3nn is specifically designed for handling data scenarios with symmetry, especially suitable for structures such as molecules and crystals that contain symmetric properties like rotation and translation. It provides a complete set of equivariant neural network construction tools that can accurately capture the symmetry characteristics of data. Researchers can efficiently develop models that conform to symmetry principles, significantly improving task performance in fields such as molecular simulation and materials science.

- Physics-Informed Neural Networks: The Physics-Informed Neural Networks library PINNs is specifically designed for integrating physical laws with data-driven modeling, especially suitable for solving partial differential equation (PDE) scenarios such as fluid mechanics and heat conduction. MindScience's pde module provides a complete toolchain for embedding physical conservation laws, boundary conditions, and other constraints into neural networks. Researchers can efficiently build models that conform to physical laws without relying on large amounts of labeled data, greatly improving solution efficiency and generalization capabilities in fields such as engineering simulation and scientific computing.

- Differentiable Solvers: Differentiable solvers are specialized tools that integrate numerical computing with automatic differentiation capabilities, primarily used for efficiently solving scientific and engineering problems such as ordinary differential equations (ODE) and partial differential equations (PDE). They can output numerical solutions while backpropagating gradient information, perfectly compatible with the MindSpore deep learning framework. With this feature, researchers can embed the solving process of physical equations into neural network training workflows, providing key support for the combination of scientific computing and deep learning, widely used in AI4Science scenarios such as dynamics simulation and control strategy optimization.

- Scientific Computing Operators: Scientific computing operators are core foundational components that support AI4Science model construction. The operators are deeply optimized in combination with Ascend to achieve efficient execution of complex scientific computing tasks.

- Distributed Parallelism: Distributed parallelism is a core efficient computing solution for handling large-scale AI4Science tasks. By splitting massive data, complex models, or computing tasks across multi-node, multi-device clusters for collaborative processing, it breaks through the computing power and memory bottlenecks of single devices, providing key computing power guarantees for AI4Science to expand toward more complex and refined research directions.

Applications
-----------

.. raw:: html

    <details>
    <summary><strong>MindChem</strong></summary>

.. list-table::
   :header-rows: 1

   * - Application Name
     - Description
     - Learning Type
   * - `diffcsp </MindChem/applications/diffcsp>`__
     - Crystal Structure Prediction
     - Supervised Learning
   * - `nequip </MindChem/applications/nequip>`__
     - Molecular Dynamics Modeling
     - Supervised Learning
   * - `orb </MindChem/applications/orb>`__
     - Molecular Orbital Analysis
     - Supervised Learning
   * - `matformer </MindChem/applications/matformer>`__
     - Material Property Prediction
     - Supervised Learning
   * - `deephe3nn </MindChem/applications/deephe3nn>`__
     - Molecular Simulation
     - Supervised Learning
   * - `crystalflow </MindChem/applications/crystalflow>`__
     - Crystal Structure Generation
     - Unsupervised Learning

.. raw:: html

    </details>

.. raw:: html

    <details>
    <summary><strong>MindEarth</strong></summary>

.. list-table::
   :header-rows: 1

   * - Application Name
     - Description
     - Learning Type
   * - `Leadformer </MindEarth/applications/leadformer>`__
     - Earth Science Time Series Prediction
     - Supervised Learning

.. raw:: html

    </details>

.. raw:: html

    <details>
    <summary><strong>MindEnergy</strong></summary>

.. list-table::
   :header-rows: 1

   * - Application Name
     - Description
     - Learning Type
   * - `DAE-PINN </MindEnergy/applications/dae_pinn>`__
     - Physics-Informed Neural Networks
     - Supervised + Physics Constraints
   * - `DeepONet-Grig-UQ </MindEnergy/applications/deeponet_grig_uq>`__
     - Functional Learning and Uncertainty Quantification
     - Supervised Learning
   * - `PowerFlowNet </MindEnergy/applications/powerflownet>`__
     - Power Flow Modeling
     - Supervised Learning

.. raw:: html

    </details>

.. raw:: html

    <details>
    <summary><strong>MindFlow</strong></summary>

.. list-table::
   :header-rows: 1

   * - Application Name
     - Description
     - Learning Type
   * - `acoustic </MindFlow/applications/acoustic>`__
     - Acoustic Simulation
     - Supervised Learning
   * - `p2c2net </MindFlow/applications/p2c2net>`__
     - Physics Modeling
     - Supervised Learning
   * - `fno2d </MindFlow/applications/fno2d>`__
     - Fourier Neural Operator
     - Supervised Learning

.. raw:: html

    </details>

.. raw:: html

    <details>
    <summary><strong>MindSPONGE</strong></summary>

.. list-table::
   :header-rows: 1

   * - Application Name
     - Description
     - Learning Type
   * - `AF3 </MindSPONGE/applications/af3>`__
     - Protein Structure Prediction
     - Supervised Learning
   * - `proteinmpnn </MindSPONGE/applications/proteinmpnn>`__
     - Protein Sequence Design
     - Supervised Learning
   * - `rfdiffusion </MindSPONGE/applications/rfdiffusion>`__
     - Protein Diffusion Generation
     - Diffusion Model/Generative Learning
   * - `protenix </MindSPONGE/applications/protenix>`__
     - Protein Function Prediction
     - Supervised Learning

.. raw:: html

    </details>

.. toctree::
   :maxdepth: 1
   :caption: Component Introduction
   :hidden:

   MindChem
   MindEarth
   MindEnergy
   MindFlow
   MindSPONGE

.. toctree::
   :maxdepth: 1
   :caption: Installation Guide
   :hidden:

   install.md

.. toctree::
   :maxdepth: 1
   :caption: Quick Start
   :hidden:

   quick_start.md

.. toctree::
   :maxdepth: 1
   :caption: API Reference
   :hidden:

   mindscience.common
   mindscience.data
   mindscience.diffuser
   mindscience.distributed
   mindscience.e3nn
   mindscience.gnn
   mindscience.models
   mindscience.pde
   mindscience.sciops
   mindscience.solvers
   mindscience.utils

.. toctree::
   :maxdepth: 1
   :caption: Contribution Guide
   :hidden:

   CONTRIBUTING

.. toctree::
   :maxdepth: 1
   :caption: RELEASE NOTES
   :hidden:

   RELEASE

