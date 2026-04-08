# Excited States and Spectroscopy Calculations

## Basic TDDFT Settings

```
! wB97X-D def2-TZVP TDDFT RIJCOSX

%tddft
  nroots 10      # Number of excited states
  triplets false  # Do not calculate triplets
end
```

## UV-Vis Spectroscopy

```
! wB97X-D def2-TZVP TDDFT RIJCOSX

%tddft
  nroots 20
end

%output
  print[P_Mayer] 1
end
```

## Electronic Circular Dichroism (ECD)

```
! wB97X-D def2-TZVP TDDFT RIJCOSX

%tddft
  nroots 20
  dotddft true
end
```

## Excited State Optimization

```
! wB97X-D def2-TZVP TDDFT OPT

%tddft
  nroots 5
  iroot 1    # Optimize the 1st excited state
end
```

## Spin-Flip TDDFT

```
! B3LYP def2-TZVP TDDFT

%tddft
  nroots 10
  sfddft true
end
```

## Output Analysis

- Excitation energies
- Oscillator strengths
- Orbital transition contributions
- Charge transfer analysis

## Functional Selection

| Application | Recommended Functional |
|-------------|------------------------|
| Valence excitations | B3LYP, PBE0 |
| Charge transfer | wB97X-D, CAM-B3LYP |
| Rydberg states | wB97X-D |
| Double excitations | NEVPT2, CASPT2 |