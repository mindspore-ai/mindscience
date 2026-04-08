# Geometry Optimization Strategies

## Basic Optimization

```
! B3LYP def2-SVP OPT
```

## Optimization + Frequency

```
! B3LYP def2-SVP OPT FREQ
```

## Optimization Settings

```
%geom
  MaxIter 100        # Maximum iterations
  Trust 0.2          # Trust radius
  Constraints        # Constraints
    { B 1 2 1.5 C }  # Bond length constraint
    { A 1 2 3 120 C }# Bond angle constraint
    { D 1 2 3 4 180 C } # Dihedral constraint
  end
end
```

## Transition State Optimization

```
! B3LYP def2-SVP OptTS FREQ

%geom
  Calc_Hess true    # Initial Hessian
end
```

## NEB-TS Method

```
! B3LYP def2-SVP NEB-TS

* xyzfile 0 1 reactant.xyz *
****
* xyzfile 0 1 product.xyz *
```

## Convergence Criteria

| Criterion | Energy | Gradient | Displacement |
|-----------|--------|----------|--------------|
| Loose | 1e-4 | 3e-3 | 6e-3 |
| Normal | 5e-5 | 1e-3 | 2e-3 |
| Tight | 1e-6 | 3e-4 | 6e-4 |

```
%geom
  Convergence Tight
end
```

## Common Issues

### Optimization Not Converging
1. Check initial structure
2. Use smaller trust radius
3. Increase iteration count
4. Try different optimizer

### Converging to Wrong Structure
1. Check symmetry
2. Use constraints
3. Start from different initial structure