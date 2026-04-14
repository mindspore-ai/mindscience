# CP2K DFT Method Selection

## Functional Selection

### GGA Functionals
| Functional | Characteristics | Use Cases |
|------|------|----------|
| PBE | Most common GGA | Solids, surfaces, molecules |
| BLYP | Molecular systems | Organic molecules |
| PW91 | Transition metals | Metal surfaces |

### Hybrid Functionals
| Functional | Characteristics | Use Cases |
|------|------|----------|
| PBE0 | 25% HF exchange | Accurate band gaps |
| B3LYP | Molecular chemistry | Organic reactions |
| HSE06 | Screened hybrid | Semiconductors |

### Van der Waals Correction
```
&XC
  &XC_FUNCTIONAL PBE
  &END XC_FUNCTIONAL
  &VDW_POTENTIAL
    POTENTIAL_TYPE PAIR_POTENTIAL
    &PAIR_POTENTIAL
      TYPE DFTD3
      PARAMETER_FILE dftd3.dat
    &END PAIR_POTENTIAL
  &END VDW_POTENTIAL
&END XC
```

## Basis Set Selection

### MOLOPT Basis Sets (Recommended)
| Basis Set | Accuracy | Use Cases |
|------|------|----------|
| SZV-MOLOPT-SR-GTH | Low | Quick preview |
| DZVP-MOLOPT-SR-GTH | Medium | Routine calculations |
| TZVP-MOLOPT-SR-GTH | High | Precise calculations |

### Pseudopotentials
| Pseudopotential | Description |
|------|------|
| GTH-PBE | PBE functional pseudopotential |
| GTH-BLYP | BLYP functional pseudopotential |
| GTH-PBE0 | Hybrid functional pseudopotential |

## Cutoff Energy Selection

| System Type | Recommended CUTOFF |
|----------|-------------|
| Molecules | 300-400 Ry |
| Solids | 400-600 Ry |
| Metals | 500-800 Ry |

## k-point Settings

### Molecular Systems
```
&KPOINTS
  SCHEME GAMMA
&END KPOINTS
```

### Periodic Systems
```
&KPOINTS
  SCHEME MONKHORST-PACK
  SIZE 4 4 4
&END KPOINTS
```