---
name: coolprop
description: High-accuracy thermophysical properties library for 122 fluids including pure fluids, pseudo-pure fluids, mixtures, and incompressible fluids. Use when calculating fluid properties, thermodynamic states, transport properties, mixture properties, or working with humid air. Supports Python, C++, MATLAB, Excel, and many other interfaces.
---

# CoolProp

CoolProp is a high-accuracy thermophysical properties library for fluids and mixtures.

## Quick Start

### Basic Property Calculation

```python
from CoolProp.CoolProp import PropsSI

# Density of Nitrogen at 298.15 K and 101325 Pa
rho = PropsSI('D', 'T', 298.15, 'P', 101325.0, 'Nitrogen')
```

### Saturation Properties

```python
# Saturation temperature of Water at 1 atm
T_sat = PropsSI('T', 'P', 101325.0, 'Q', 0.0, 'Water')

# Saturated vapor enthalpy
H_v = PropsSI('H', 'P', 101325.0, 'Q', 1.0, 'Water')

# Saturated liquid enthalpy
H_l = PropsSI('H', 'P', 101325.0, 'Q', 0.0, 'Water')

# Latent heat of vaporization
H_fg = H_v - H_l
```

### Mixture Properties

```python
# Air mixture at 1 atm and 300 K
rho = PropsSI('D', 'P', 101325.0, 'T', 300.0, 'Air.mix')
```

## Core Concepts

### Supported Fluids

CoolProp includes 122 predefined fluids:
- Pure fluids (Water, Air, Nitrogen, CO2, etc.)
- Pseudo-pure fluids (R134a, R125, Refrigerants)
- Mixtures (Air, combustion gases, refrigerants)
- Incompressible fluids

### Property Types

**Thermodynamic properties**: T, P, D, H, S, U, G, A, Cp, Cv
**Transport properties**: viscosity, conductivity, Prandtl number, surface tension
**Derived properties**: speed of sound, compressibility factor, enthalpy of vaporization
**Critical properties**: Tcrit, Pcrit, rhocrit, etc.
**Humidity properties**: relative humidity, dew point, wet bulb temperature

### Input Pairs

CoolProp uses input pairs to define state:
- `('T', 'P')`: Temperature and pressure (fastest)
- `('P', 'T')`: Pressure and temperature (slower)
- `('D', 'T')`: Density and temperature
- Other combinations available

### Units

CoolProp uses SI units throughout:
- Temperature: K
- Pressure: Pa
- Density: kg/m³
- Enthalpy: J/mol or J/kg
- Entropy: J/mol/K or J/kg/K
- Viscosity: Pa·s
- Conductivity: W/m/K

## Fluid Categories

### Pure and Pseudo-Pure Fluids

**Use case**: Single-component thermodynamic calculations

**Characteristics**:
- Equation of state for pure components
- High accuracy reference data
- Wide temperature and pressure ranges
- Transport properties available

**Examples**:
```python
# Water properties
rho_water = PropsSI('D', 'T', 300.0, 'P', 101325.0, 'Water')
cp_water = PropsSI('Cp', 'T', 300.0, 'P', 101325.0, 'Water')

# Refrigerant properties
rho_r134a = PropsSI('D', 'T', 250.0, 'P', 101325.0, 'R134a')
cp_r134a = PropsSI('Cp', 'T', 250.0, 'P', 101325.0, 'R134a')
```

### Mixtures

**Use case**: Multi-component fluid systems

**Characteristics**:
- High-accuracy Helmholtz energy formulation
- Binary interaction parameters
- Phase envelope calculations
- Excess Helmholtz energy terms

**Examples**:
```python
# Predefined mixture (Air)
rho_air = PropsSI('D', 'P', 101325.0, 'T', 300.0, 'Air.mix')

# User-defined mixture
# Propane/Ethane mixture
rho_mix = PropsSI('D', 'P', 101325.0, 'T', 300.0, 
                  'HEOS::R32[0.697615]&Ethane[0.302385]')
```

### Incompressible Fluids

**Use case**: Liquid properties at specified conditions

**Characteristics**:
- Partial derivatives available
- Fitting reports for property correlations
- Transport property correlations

**Examples**:
```python
# Water incompressible properties
cp_water = PropsSI('Cp', 'T', 300.0, 'P', 101325.0, 'Water')
visc_water = PropsSI('viscosity', 'T', 300.0, 'P', 101325.0, 'Water')
```

### Humid Air

**Use case**: Psychrometric calculations and humidity

**Characteristics**:
- High-accuracy humidity calculations
- Wet bulb and dew point calculations
- Enhancement factor for moist air
- Isothermal and isobaric properties

**Examples**:
```python
from CoolProp.CoolProp import HumidAir

# Humid air properties
ha = HumidAir(298.15, 101325.0, 0.5)  # T, P, RH
W_bulb = ha.W_bulb()  # Wet bulb temperature
T_dp = ha.T_dp()        # Dew point temperature
```

### IF97 Steam/Water

**Use case**: Industrial steam and water calculations

**Characteristics**:
- IAPWS-IF97 industrial formulation
- Fast execution
- Valid for specific ranges
- High accuracy for steam properties

**Examples**:
```python
# IF97 water properties
rho_water = PropsSI('D', 'T', 500.0, 'P', 10e6, 'IF97::Water')

# IF97 steam properties
h_steam = PropsSI('H', 'T', 500.0, 'P', 10e6, 'IF97::Water')
```

## Advanced Features

### Phase Determination

```python
from CoolProp.CoolProp import PhaseSI

# Determine phase at given conditions
phase = PhaseSI('P', 101325.0, 'Q', 0.0, 'Water')
# Returns: 'liquid', 'vapor', 'twophase', etc.
```

### Partial Derivatives

```python
# First partial derivatives
dH_dT = PropsSI('d(H)/d(T)|P', 'P', 101325.0, 'T', 300.0, 'Water')

# Second partial derivatives
d2H_dT2 = PropsSI('d(d(H)/d(T)|P)/d(H)/d(T)|P', 
                  'P', 101325.0, 'T', 300.0, 'Water')
```

### Reference States

```python
# Set reference state for entropy calculations
from CoolProp.CoolProp import set_reference_stateS

set_reference_stateS('n-Propane', 'ASHRAE')

# Now entropy calculations use reference state
s = PropsSI('S', 'T', 300.0, 'P', 101325.0, 'n-Propane')
```

### REFPROP Integration

```python
# Use REFPROP library if available
rho = PropsSI('D', 'T', 300.0, 'P', 101325.0, 'REFPROP::Water')
```

## Resources

- **Pure fluids**: See [pure_fluids.md](references/pure_fluids.md)
- **Mixtures**: See [mixtures.md](references/mixtures.md)
- **Humid air**: See [humid_air.md](references/humid_air.md)
- **Incompressible fluids**: See [incompressible.md](references/incompressible.md)
- **Advanced features**: See [advanced.md](references/advanced.md)
- **Interfaces**: See [interfaces.md](references/interfaces.md)
