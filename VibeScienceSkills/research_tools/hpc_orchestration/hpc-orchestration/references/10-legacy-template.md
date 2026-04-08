## HPC Agent Skill: [Software Name]

## 1. Capabilities & Scope
**Target Software**: [Software Name] (e.g., OpenFOAM, LAMMPS, FEniCS)
**Applicable Physics Fields**: [Applicable domains, e.g., incompressible flow, molecular dynamics tensile testing, etc.]
**Core Capabilities**:
- [Capability 1, e.g., automatic generation of blockMeshDict and controlDict]
- [Capability 2, e.g., automatic correction of timestep divergence errors]

## 2. Dependencies & Environment
- **Commands**: [e.g., `mpirun`, `simpleFoam`, `lmp`]
- **Test Command**: [e.g., `[Software Name] -version` to check installation]
- **File Extensions**: [e.g., `.in`, `.txt`, `.dict`]

## 3. Standardized Workflow

As a CLI Agent, when a user requests to use this software, **strictly follow the sequence below**:

### Phase 1: Pre-processing
1. **Task**: Generate or read geometry mesh/initial topology.
2. **Knowledge Association**: [Specify which knowledge base files to reference, e.g., refer to the Mesh section in `KNOWLEDGE_DICT.md`]
3. **Expected Output**: [e.g., generate `constant/polyMesh/` or `data.lmp`]

### Phase 2: Configuration Generation
1. **Task**: Generate core input files based on user physics requirements.
2. **Mandatory Rules**:
   - Must read and follow "required fields" and "constraints" in `KNOWLEDGE_DICT.md`.
   - Must reference `KNOWLEDGE_PHYSICS.md` to select correct physics model parameters.
3. **Expected Output**: [e.g., generate various dictionary files under `system/`]

### Phase 3: Execution & Self-Healing
1. **Task**: Start computation process or submit to HPC queue.
2. **Tool Invocation**:
   - Use `execute` for local runs.
   - For HPC cluster runs, invoke global tool `hpc_slurm_deploy.py` to generate scripts.
3. **Self-Healing Mechanism**:
   - If terminal returns Error Code != 0, **strictly forbidden to guess the cause**.
   - Must immediately read `KNOWLEDGE_ERRORS.md`.
   - Use regex or keywords to match Error ID in standard error (stderr).
   - Strictly follow the Agent Action corresponding to Error ID to modify configuration files and retry.

### Phase 4: Post-processing & Result Summary
1. **Task**: Extract scalar results, monitor residual convergence, or generate visualization post-processing slices.
2. **Tool Invocation**: Use post-processing scripts under `scripts/` bundled with this Skill (e.g., `post_paraview.py`).
3. **Expected Output**: Output final convergence information or key physics quantity summary to user.

## 4. Knowledge Files Reference
When executing this Skill, please call `read_file` at any time to read the following specifications:
- [ ] `KNOWLEDGE_DICT.md`: Configuration syntax and constraints
- [ ] `KNOWLEDGE_PHYSICS.md`: Physics field empirical rules
- [ ] `KNOWLEDGE_ERRORS.md`: Exception self-healing guide
- [ ] `TEMPLATES/`: Basic configuration template directory