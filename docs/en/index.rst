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
----------------

.. raw:: html

    <details>
    <summary><strong>MindChem</strong></summary>

.. list-table::
   :header-rows: 1

   * - Application Name
     - Description
     - Learning Type
   * - `DiffCSP <https://atomgit.com/mindspore-lab/mindscience/blob/master/MindChem/applications/diffcsp>`__
     - Crystal Structure Generation
     - Diffusion Model/Generative Learning
   * - `NequIP <https://atomgit.com/mindspore-lab/mindscience/blob/master/MindChem/applications/nequip>`__
     - Molecular Force Field
     - Supervised Learning
   * - `Orb <https://atomgit.com/mindspore-lab/mindscience/blob/master/MindChem/applications/orb>`__
     - Molecular Force Field
     - Supervised Learning
   * - `MatFormer <https://atomgit.com/mindspore-lab/mindscience/blob/master/MindChem/applications/matformer>`__
     - Material Property Prediction
     - Supervised Learning
   * - `DeepHE3nn <https://atomgit.com/mindspore-lab/mindscience/blob/master/MindChem/applications/deephe3nn>`__
     - Hamiltonian Prediction
     - Supervised Learning
   * - `CrystalFlow <https://atomgit.com/mindspore-lab/mindscience/blob/master/MindChem/applications/crystalflow>`__
     - Crystal Structure Generation
     - Diffusion Model/Generative Learning

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
   * - `LeadFormer <https://atomgit.com/mindspore-lab/mindscience/blob/master/MindEarth/applications/leadformer>`__
     - Arctic Sea Ice Lead Prediction
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
   * - `DAE-PINN <https://atomgit.com/mindspore-lab/mindscience/blob/master/MindEnergy/applications/dae_pinn>`__
     - Physics-Informed Neural Networks
     - Supervised + Physics Constraints
   * - `DeepONet-Grig-UQ <https://atomgit.com/mindspore-lab/mindscience/blob/master/MindEnergy/applications/deeponet_grig_uq>`__
     - Post-Fault Trajectory Prediction in Power Grid
     - Supervised Learning
   * - `PowerFlowNet <https://atomgit.com/mindspore-lab/mindscience/blob/master/MindEnergy/applications/powerflownet>`__
     - Power Flow Calculation
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
   * - `Acoustic <https://atomgit.com/mindspore-lab/mindscience/blob/master/MindFlow/applications/acoustic>`__
     - Acoustic Simulation
     - Auto-Iterative Learning
   * - `P2C2Net <https://atomgit.com/mindspore-lab/mindscience/blob/master/MindFlow/applications/p2c2net>`__
     - Physics Modeling for Solving 2D Burgers Equations
     - Supervised Learning
   * - `FNO2D <https://atomgit.com/mindspore-lab/mindscience/blob/master/MindFlow/applications/fno2d>`__
     - Fourier Neural Operator for Solving 2D Navier-Stokes Equations
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
   * - `AlphaFold3 <https://atomgit.com/mindspore-lab/mindscience/blob/master/MindSPONGE/applications/af3>`__
     - Protein Structure Prediction
     - Supervised Learning
   * - `ProteinMPNN <https://atomgit.com/mindspore-lab/mindscience/blob/master/MindSPONGE/applications/proteinmpnn>`__
     - Protein Sequence Design
     - Supervised Learning
   * - `RFdiffusion <https://atomgit.com/mindspore-lab/mindscience/blob/master/MindSPONGE/applications/rfdiffusion>`__
     - Protein Diffusion Generation
     - Diffusion Model/Generative Learning
   * - `Protenix <https://atomgit.com/mindspore-lab/mindscience/blob/master/MindSPONGE/applications/protenix>`__
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

   api/mindscience.common
   api/mindscience.data
   api/mindscience.diffuser
   api/mindscience.distributed
   api/mindscience.e3nn
   api/mindscience.gnn
   api/mindscience.models
   api/mindscience.pde
   api/mindscience.sciops
   api/mindscience.solvers
   api/mindscience.utils

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

