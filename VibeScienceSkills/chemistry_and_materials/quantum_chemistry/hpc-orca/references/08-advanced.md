# Advanced Features

## ONIOM (QM/MM)

```
! B3LYP def2-SVP ONIOM

%oniom
  qm_region 1-10
  mm_region 11-100
end
```

## Relativistic Corrections

### ZORA

```
! B3LYP def2-SVP ZORA
```

### DKH

```
! B3LYP def2-SVP DKH2
```

Suitable for: heavy element calculations

## Open-Shell Systems

```
! UHF def2-SVP
* xyz 0 2    # Multiplicity = 2
[coordinates]
*
```

## Spin State Optimization

```
! UKS B3LYP def2-SVP

%scf
  spinflip true
end
```

## Broken Symmetry

```
! UKS B3LYP def2-SVP

%scf
  brokenSym 1,2  # Atoms 1 and 2
end
```

## Localized Orbital Analysis

```
%output
  print[P_Loewdin] 1
  print[P_Mayer] 1
end
```

## NBO Analysis

```
! B3LYP def2-SVP NBO
```

## Molecular Orbital Output

```
%output
  print[P_MOs] 1
end
```

## Cube File Generation

```
%plots
  format cube
  MO(1-10)
end
```