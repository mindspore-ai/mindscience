# Error Recovery Strategies

## SCF Not Converging

### Symptoms
```
WARNING: SCF DID NOT CONVERGE!
```

### Solutions

1. Increase iteration count
```
%scf
  MaxIter 500
end
```

2. Use damping
```
%scf
  damping 0.2
end
```

3. Use level shift
```
%scf
  shift shift 0.1 0.1
end
```

4. Relax convergence criteria
```
%scf
  convergence normal
end
```

5. Change initial guess
```
%scf
  guess pmodel
end
```

## Geometry Optimization Not Converging

### Symptoms
```
GEOMETRY OPTIMIZATION DID NOT CONVERGE
```

### Solutions

1. Increase iteration count
```
%geom
  MaxIter 200
end
```

2. Use smaller trust radius
```
%geom
  Trust 0.1
end
```

3. Calculate initial Hessian
```
%geom
  Calc_Hess true
end
```

4. Use different optimizer
```
%geom
  optimizer bfgs
end
```

## Insufficient Memory

### Symptoms
```
MEMORY ALLOCATION ERROR
```

### Solutions

1. Increase memory per core
```
%maxcore 8000
```

2. Reduce number of parallel cores
```
%pal nprocs 4 end
```

3. Use RI approximation to reduce memory
```
! RIJCOSX
```

## Basis Set Error

### Symptoms
```
BASIS SET NOT FOUND
```

### Solutions

1. Check basis set name spelling
2. Use standard basis sets
3. Manually specify basis set file

## Parallel Error

### Symptoms
```
MPI ERROR
```

### Solutions

1. Check OpenMPI version
2. Ensure using shared version
3. Check environment variables

## File Corruption

### Solutions

1. Restart from checkpoint
```
! MOREAD
%moinp "backup.gbw"
```

2. Use last geometry
```
* xyzfile 0 1 orca.xyz *
```