# NWChem Parallel Computing

## MPI Parallel

```bash
mpirun -np 8 nwchem input.nw
```

## SLURM Script

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32

mpirun -np $SLURM_NTASKS nwchem input.nw
```

## Memory Settings

```
memory total 16 gb
memory heap 4 gb
memory stack 4 gb
```

## Parallel Strategy

| System Size | Recommended Configuration |
|-------------|---------------------------|
| Small molecules | 1 node, 8 cores |
| Medium systems | 1 node, 32 cores |
| Large systems | Multi-node, 64+ cores |

## Performance Optimization

1. Use appropriate basis sets
2. Adjust memory allocation
3. Use density fitting
4. Optimize MPI configuration