# CP2K Molecular Dynamics

## AIMD (Born-Oppenheimer MD)

```
&MOTION
  &MD
    ENSEMBLE NVT
    STEPS 10000
    TIMESTEP 0.5
    TEMPERATURE 300
    
    &THERMOSTAT
      TYPE CSVR
      TIMECON 100
    &END THERMOSTAT
  &END MD
&MOTION
```

## CPMD (Car-Parrinello MD)

```
&MOTION
  &MD
    ENSEMBLE NVT
    STEPS 10000
    TIMESTEP 0.1
    TEMPERATURE 300
    
    &THERMOSTAT
      TYPE NOSE_HOOVER
      NOSE_HOOVER_CHAINLENGTH 3
    &END THERMOSTAT
  &END MD
&MOTION
```

## Ensemble Types

| Ensemble | Description | Conserved Quantity |
|------|------|--------|
| NVE | Microcanonical | Energy |
| NVT | Canonical | Temperature |
| NPT_I | Isothermal-isobaric (isotropic) | P, T |
| NPT_F | Isothermal-isobaric (anisotropic) | P, T |

## Thermostats

| Type | Characteristics |
|------|------|
| CSVR | Smooth, recommended |
| NOSE_HOOVER | Classic method |
| LANGEVIN | Stochastic method |

## Barostats

```
&BAROSTAT
  PRESSURE 1.0
  TIMECON 100
&END BAROSTAT
```

## Output Control

```
&PRINT
  &TRAJECTORY
    FORMAT XYZ
    EACH 100
  &END TRAJECTORY
  &VELOCITIES ON
  &END VELOCITIES
  &FORCES ON
  &END FORCES
&END PRINT
```