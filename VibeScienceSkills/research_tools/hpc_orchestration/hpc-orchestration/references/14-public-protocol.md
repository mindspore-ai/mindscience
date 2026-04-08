## HPC Skills Public Protocol

## Purpose

This reference defines the repository-level contract for portable HPC skills so that solver-specific skills and `hpc-orchestration` interoperate cleanly.

## Canonical skill shape

Each maintained skill should use this layout:

```text
skills/<skill-name>/
|- SKILL.md
|- agents/openai.yaml
|- references/
|- assets/templates/
\- scripts/                # optional
```

Rules:

- `skills/` is the only maintained distribution surface
- `SKILL.md` stays concise and procedural
- detailed manuals and matrices live in `references/`
- reusable scaffolds live in `assets/templates/`
- deterministic helper tooling lives in `scripts/`

## Loading contract

Use progressive disclosure.

1. read `SKILL.md` first
2. load only the referenced materials needed for the current task
3. reuse templates before regenerating boilerplate
4. call bundled scripts when a deterministic path already exists

Anti-patterns:

- loading every reference file at once
- burying essential workflow rules only inside large reference files
- duplicating the same manual content in `SKILL.md` and `references/`

## Responsibility split

Solver-specific skills own:

- domain inputs
- physics and numerics
- solver-specific post-processing
- solver-specific failure classification and repair guidance

`hpc-orchestration` owns:

- scheduler choice
- queue submission scaffolds
- launch patterns
- environment and storage hygiene
- monitoring and lifecycle control
- cluster-level debugging, profiling, and handoff patterns

## Workflow contract

Repository-wide HPC work should be expressible as:

1. stage inputs
2. generate configuration
3. validate a small run
4. submit or launch through the scheduler
5. monitor queue and logs
6. classify failures
7. repair inputs or runtime shape
8. resume, restart, or post-process

Every solver skill does not need to implement all eight stages directly, but it should be compatible with this lifecycle.

## Reference design rules

Prefer distinct reference files by concern:

- one full manual for orientation
- specialized references for launch, storage, containers, remote development, and profiling
- matrix or dictionary references where compact decision support is better than prose

Keep references one hop away from `SKILL.md` whenever possible.

## Template design rules

Templates should be:

- portable by default
- explicit about placeholders
- safe for Linux-centric HPC environments
- narrow enough that an agent can adapt them with small edits

Do not hardcode:

- local absolute paths
- one organization's account strings
- site-specific module names unless the template is explicitly marked as site-local

## Path and platform rules

Portable documentation should prefer:

- repository-relative paths for files inside this repo
- Linux-style examples for HPC command lines
- `$HOME`, `$PROJECT`, `$SCRATCH`, or generic variables instead of machine-local absolute paths

Avoid Windows-only path examples inside skill content unless the skill explicitly targets Windows.

## Validation expectations

Before treating a skill revision as complete:

1. confirm the maintained shape still holds
2. check that new references are directly discoverable from `SKILL.md`
3. verify templates live under `assets/templates/`
4. run the available skill validator when possible

## Open-source hygiene

Do not commit:

- personal absolute paths
- private cluster hostnames unless intentionally public
- local caches, outputs, or heavyweight binary artifacts

Portable skills should assume the cluster is Linux-first and multi-user.
