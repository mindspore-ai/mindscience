# Software Build And Reproducibility

## Purpose

Use this reference when the cluster workflow needs compilation, package installation, environment pinning, or a reproducible rebuild path.

## Preferred environment ladder

Choose the simplest viable layer first:

1. site-provided modules
2. site-supported package manager or environment stack
3. container image
4. user-local source build

The lower you go on this ladder, the more operational burden you own.

## Build placement

Portable defaults:

- light compilation may be acceptable on login nodes if the site allows it
- heavy parallel builds should use an interactive or batch allocation
- large build trees should not silently fill a small home quota

If unsure, treat a parallel build as a scheduled workload.

## Compiler and MPI consistency

Keep these aligned:

- compiler family
- MPI family
- math libraries
- application modules

Do not mix stacks casually. A solver built with one MPI family may not run correctly with another at launch time.

## Capturing reproducibility

For every important build or environment, record:

- module list
- compiler and MPI versions
- configure or CMake command
- install prefix
- git revision or source archive identifier
- runtime executable path

Store this next to the run metadata or in a build note committed with the workflow when appropriate.

## CMake and Make habits

Recommended habits:

- use out-of-source builds
- keep one build directory per toolchain
- version the configure command
- avoid editing generated cache files manually unless necessary

Portable example:

```bash
cmake -S . -B build-gcc-openmpi -DCMAKE_BUILD_TYPE=Release
cmake --build build-gcc-openmpi -j 8
```

## Spack or environment managers

When a site supports a package manager such as Spack:

- prefer a site-provided stack first
- use named environments for project-scoped reproducibility
- export the environment specification if the workflow depends on it

## Python environments

For Python-heavy tooling:

- activate environments inside the batch or interactive script
- pin major dependencies when reproducibility matters
- keep package caches under control
- consider a container when the environment is too fragile for repeated ad hoc repair

## Containers as reproducibility boundary

Containers are often the cleanest way to pin a userland for:

- preprocessing pipelines
- post-processing stacks
- Python analytics
- application dependencies that are hard to rebuild repeatedly

Use [container-workflows.md](container-workflows.md) when the build problem becomes an environment portability problem.

## Failure patterns

| Symptom | Likely cause | First repair |
| --- | --- | --- |
| build succeeds but runtime launcher fails | compiler or MPI stack mismatch | rebuild or run with a consistent stack |
| package works interactively but not in batch | activation only happens in the shell session | move activation into the job script |
| home quota fills during install | build tree or caches in the wrong place | move builds and caches to project or scratch storage |
| repeated manual environment repair | stack is too fragile | pin it with modules, Spack environment, or a container |
