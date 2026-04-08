# ORCA Input File Structure

## Basic Structure

```
! [keyword] [keyword] ...

%[module_name]
  setting value
end

* [coordinate_format] [charge] [multiplicity]
[atomic_coordinates]
*
```

## Keyword Line

Starts with `!`, specifies calculation type, method, basis set, etc.:

```
! B3LYP def2-SVP OPT FREQ RIJCOSX TightSCF
```

### Common Keywords

| Category | Keyword | Description |
|----------|---------|-------------|
| Calculation Type | SP | Single point energy calculation |
| | OPT | Geometry optimization |
| | FREQ | Frequency calculation |
| | OPT FREQ | Optimization + frequency |
| | TDDFT | Excited state calculation |
| Method | B3LYP | Hybrid functional |
| | PBE0 | PBE0 functional |
| | wB97X-D | Range-separated functional + dispersion correction |
| | M06-2X | Meta-GGA functional |
| | MP2 | Second-order perturbation theory |
| Basis Set | def2-SVP | Double-zeta polarized basis set |
| | def2-TZVP | Triple-zeta polarized basis set |
| | def2-QZVP | Quadruple-zeta polarized basis set |
| | cc-pVTZ | Correlation-consistent basis set |
| Acceleration | RIJCOSX | RI-J + COSX acceleration |
| | RI-J | RI-J acceleration |
| | DLPNO | Domain-based Local Pair Natural Orbital |
| Convergence | TightSCF | Tight SCF convergence |
| | VeryTightSCF | Very tight SCF convergence |

## Module Settings

### Memory and Parallelization

```
%maxcore 4000        # Memory per core (MB)
%pal nprocs 8 end    # Number of parallel cores
```

### SCF Settings

```
%scf
  MaxIter 200
  convergence tight
  shift shift 0.1 0.1
end
```

### Geometry Optimization

```
%geom
  MaxIter 100
  Trust 0.2
  Constraints
    { B 1 2 1.5 }
  end
end
```

### TDDFT Settings

```
%tddft
  nroots 10
  triplets false
  tda false
end
```

## Coordinate Input

### Embedded Coordinates

```
* xyz 0 1
O  0.000  0.000  0.117
H  0.000  0.757 -0.469
H  0.000 -0.757 -0.469
*
```

### External File

```
* xyzfile 0 1 molecule.xyz *
```

### Gaussian Format

```
* gjf 0 1
O
H 1 0.96
H 1 0.96 2 104.5
*
```

## Output Files

| File | Content |
|------|---------|
| .out | Main output file |
| .xyz | Final geometry structure |
| .gbw | Basis set/wavefunction |
| .hess | Hessian matrix |
| .cis | TDDFT results |
| .opt | Optimization trajectory |