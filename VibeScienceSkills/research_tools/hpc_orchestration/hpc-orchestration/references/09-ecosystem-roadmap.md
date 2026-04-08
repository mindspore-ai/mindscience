## HPC Agent Skills Collection: Computational Physics Automation Full Lifecycle Implementation Plan

## 1. Core Design Philosophy: Based on CLI Agent Public Skills Protocol
This plan aims to eliminate complex external RAG dependencies by solidifying all physics field and solver expertise, API syntax, code templates, and error recovery rules **into high-information-density structured Markdown files**.
CLI Agents (such as Cline, RooCode, OpenHands, or your current AI Assistant) before executing computational tasks, directly read these structured `KNOWLEDGE.md` and `RULES.md` files through tools (such as `read_file`), thereby precisely and stably generating configuration scripts, and invoking `execute` to complete automated deployment, execution, and post-processing.

---

## 2. Market Research and Reference from Similar Projects
In the field of AI and HPC integration, the industry is evolving from "hardcoded high-throughput workflows" toward "LLM-Agent autonomous driving". When designing the Skills protocol, we referenced the design patterns from the following open-source projects or platforms:
- **AiiDA / ASE**: Drawing from their unified API interface philosophy for molecular dynamics (MD) and first-principles (DFT), abstracting `VASP` and `LAMMPS` operations into Python generation Skills.
- **OpenFOAMGPT / FoamPilot**: Learning from their successful experience, the areas where Agents are most likely to fail are **supercomputer queue script syntax** and **physical quantity boundary condition orthogonality**. Therefore, this plan will enforce a valid combination dictionary for boundary conditions in the Markdown knowledge base.
- **FEniCS / Firedrake**: These software packages are themselves Python partial differential equation (PDE) solvers, naturally aligned with LLM generation capabilities, making them the best starting point for validating the "code generation" closed loop.

---

## 3. HPC Physics Simulation Software Panorama Skill Graph
To create a "collection of Skills for all HPC simulation software on the market", we classify by physics field and computational paradigm, developing an independent Skill Package for each software:

### 3.1 Core Computational Fluid Dynamics (CFD)
- **OpenFOAM**: The most widely used industrial open-source CFD software. Key Skills: dictionary generation (`blockMeshDict`, `controlDict`), parallel decomposition (`decomposeParDict`), residual monitoring.
- **SU2**: Aerodynamic optimization open-source software. Key Skills: adjoint equation optimization configuration generation, mesh deformation scripts.
- **Nek5000**: High-order spectral element method. Key Skills: Fortran code insertion, extremely high parallelism HPC deployment script generation.

### 3.2 Solid Mechanics, Multiphysics and Finite Element Method (FEM)
- **FEniCS / Firedrake**: Code is mathematics. Key Skills: directly translating natural language PDEs (such as variational forms) into Python UFL (Unified Form Language) solver scripts.
- **Elmer FEM**: Multiphysics (thermal, fluid, electromagnetic, acoustic). Key Skills: structured mapping of `sif` (Solver Input File) configuration syntax rules.
- **MOOSE**: Large-scale nuclear energy and multiphysics framework based on libMesh. Key Skills: multi-block coupled input file generation validation.
- **CalculiX**: Abaqus open-source alternative. Key Skills: `.inp` card generation (node definition, material constitutive, load step generation).

### 3.3 First-Principles and Quantum Chemistry (DFT/Quantum)
- **VASP (commercial/semi-open)**: Solid-state physics standard. Key Skills: `INCAR` (parameter control), `POSCAR` (structure), `KPOINTS` generation.
- **Quantum ESPRESSO**: Fully open-source VASP alternative. Key Skills: pseudopotential selection rules, self-consistent field (SCF) error self-healing.

### 3.4 Molecular Dynamics (MD)
- **LAMMPS**: Materials MD dominant software. Key Skills: `in.lammps` script writing, force field assignment rules, ensemble (NVE/NVT/NPT) switching.
- **GROMACS**: Biomolecular dominant software. Key Skills: topology file (`.top`) generation, energy minimization, MD production simulation workflow.

---

## 4. Standardized Skill Directory and Structured Knowledge Specification (No RAG Required)
Each HPC software as an independent Skill package follows the public Skills protocol. Below is an example using `hpc-openfoam` to demonstrate directory structure and knowledge organization:

```text
skills/hpc-openfoam/
├── SKILL.md                 # Agent entry document: defines capability boundaries, use cases, workflow sequence
├── KNOWLEDGE_DICT.md        # Core knowledge base: structured listing of OpenFOAM dictionary file syntax trees, required and optional fields
├── KNOWLEDGE_PHYSICS.md     # Physics knowledge base: e.g., required boundary files (k, epsilon, nut) and initial value formulas for RANS/LES
├── KNOWLEDGE_ERRORS.md      # Error self-healing library: lists common errors (e.g., floating point overflow, Courant Number too large) and corresponding Agent modification actions
├── TEMPLATES/               # Static template library: basic boundary conditions, HPC Slurm submission workflow Jinja/text templates
└── scripts/
    ├── log_monitor.py       # (Execution domain) Python script for real-time monitoring of computation divergence and interruption
    └── post_paraview.py     # (Post-processing domain) Script for automatically extracting lift/drag forces or slice contour plots
```

### Knowledge Base Writing Example (`KNOWLEDGE_DICT.md` excerpt)
Abandoning RAG, requiring large models to strictly read the following Markdown specifications before writing files:
```markdown
### Target File: `system/controlDict`
**Required key-value pairs and valid parameter ranges**:
- `application`: [icoFoam | simpleFoam | pimpleFoam | interFoam] (select based on physics field)
- `startFrom`: [startTime | firstTime | latestTime] (default: latestTime)
- `deltaT`: Must satisfy Courant Number < 1, suggested initial value 1e-4.
- `writeControl`: [timeStep | runTime | adjustableRunTime] (recommended: adjustableRunTime)

**Prohibited behaviors (Constraints)**:
1. Never omit the trailing semicolon `;` in dictionaries.
2. For incompressible flow (simpleFoam), do not define thermodynamic variables (such as `T`, `p_rgh`).
```

---

## 5. Project Implementation Plan (5 Phases)

### Phase 1: Protocol Definition and Infrastructure Setup (Month 1)
- **Goal**: Complete writing of "HPC CLI Agent Public Skills Protocol", establish directory structure, trigger conditions, and document loading mechanism.
- **Tasks**:
  1. Develop generic `SKILL.md` template, covering four standardized phases: **preprocessing, configuration, execution (training), post-processing**.
  2. Develop CLI-based generic tools: `hpc_slurm_deploy.py` (generic job queue generator) and `hpc_log_tracker.py` (generic residual monitor).

### Phase 2: Python Native Computational Ecology Pipeline (Month 2)
- **Goal**: Start from the area most amenable to LLM code generation, validate the "structured knowledge Markdown -> script generation -> local/container execution" closed loop.
- **Implementation targets**: `hpc-fenics` (finite element), `hpc-ase` (atomic simulation environment, covering basic VASP/LAMMPS preprocessing).
- **Deliverables**:
  1. Complete FEniCS `KNOWLEDGE_UFL.md` syntax rule writing.
  2. Implement natural language generation of complete Python scripts for solving linear elasticity or Poisson equation, and automatically extract VTK results.

### Phase 3: Hard Configuration Type Classic Software Breakthrough (Month 3-4)
- **Goal**: Tackle classic C/Fortran solvers with highly customized input files that easily crash from a single typo.
- **Implementation targets**: `hpc-openfoam`, `hpc-elmerfem`, `hpc-lammps`.
- **Tasks**:
  1. Extensively write plain text structured `KNOWLEDGE_DICT.md` and `KNOWLEDGE_PHYSICS.md`.
  2. Implement Agent's "trial-and-error self-healing" mechanism: execute `mpirun` -> read standard error output -> regex match `KNOWLEDGE_ERRORS.md` -> automatically modify configuration and resubmit.

### Phase 4: HPC Cluster Deployment and Large-Scale Scheduling (Month 5)
- **Goal**: Move beyond local single-machine Sandbox, achieve integration with real supercomputer nodes (Slurm/PBS).
- **Tasks**:
  1. Refine **Deployment & Training domain** Skills: introduce `KNOWLEDGE_SLURM.md`, guide Agent to automatically calculate optimal node count based on mesh count, generate `sbatch` scripts.
  2. Develop cross-node log monitoring logic, enabling Agent to monitor remote supercomputer queue and computation status from local CLI terminal.

### Phase 5: Post-Processing and Cross-Software Collaboration (Month 6)
- **Goal**: Achieve automated data visualization and multi-Agent collaboration for multiphysics problems.
- **Tasks**:
  1. Refine **Post-processing domain** Skills: write PyVista and ParaView trace macro syntax into `KNOWLEDGE_VISUAL.md`, enabling Agent to automatically capture contour plots, plot convergence curves, and output illustrated final Markdown/PDF experiment reports.
  2. Test cross-software collaboration capabilities (e.g., use `hpc-openfoam` to compute flow field, extract surface pressure, then pass to `hpc-calculix` for structural stress calculation).

---

## Conclusion
By solidifying expert experience into **high-density structured Markdown rules**, allowing CLI Agents to independently consult and follow these rules through tools, we not only bypass the "retrieval hallucination" and "context fragmentation" problems of RAG, but also achieve precise control over physics simulations at extremely low cost. This will be the best path to building out-of-the-box, highly robust HPC AI Agents.