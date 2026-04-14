# Thermodynamics in Cantera

Complete guide to thermodynamic properties and phase equilibrium.

## Creating Phases

### Ideal Gas Phase

```python
import cantera as ct

# From YAML file
gas = ct.Solution('gri30.yaml', 'gas')
gas.TPX = 300, 101325, 'N2:0.79, O2:0.21'

# From CTI file (legacy)
gas = ct.Solution('gas.cti', 'gas')
```

### Liquid Phase

```python
# Redlich-Kwong EOS
liquid = ct.Solution('liquid.yaml', 'liquid')
liquid.TPX = 300, 101325, 'H2O:1.0'
```

### Solid Phase

```python
# Solid phase
solid = ct.Solution('solid.yaml', 'solid')
solid.TP = 300, 101325
```

### Incompressible Liquid

```python
# Constant density liquid
liquid = ct.Solution('liquid.yaml', 'liquid')
liquid.TPX = 300, 101325, 'H2O:1.0'
```

## State Variables

### Setting State

```python
# Temperature, Pressure, Composition (TPX)
gas.TPX = 300, 101325, 'H2:1, O2:0.5, N2:0.5'

# Temperature, Density, Composition (TDX)
gas.TDX = 300, 1.0, 'H2:1, O2:0.5, N2:0.5'

# Temperature, Pressure, Mole Fractions (TPX)
gas.TPX = 300, 101325, [0.5, 0.25, 0.25]

# Enthalpy, Pressure (HP)
gas.HP = gas.enthalpy_mole, 101325

# Entropy, Pressure (SP)
gas.SP = gas.entropy_mole, 101325

# Internal Energy, Density (UV)
gas.UV = gas.int_energy_mole, 1.0
```

### Getting State

```python
T = gas.T                    # Temperature (K)
P = gas.P                    # Pressure (Pa)
D = gas.density               # Density (kg/m³)
h = gas.enthalpy_mole          # Enthalpy (J/mol)
s = gas.entropy_mole           # Entropy (J/mol/K)
g = gas.gibbs_mole             # Gibbs energy (J/mol)
u = gas.int_energy_mole         # Internal energy (J/mol)
cp = gas.cp_mole               # Heat capacity at const P (J/mol/K)
cv = gas.cv_mole               # Heat capacity at const V (J/mol/K)
```

## Thermodynamic Properties

### Molar Properties

```python
# Enthalpy
h = gas.enthalpy_mole          # J/mol
h_mass = gas.enthalpy_mass       # J/kg

# Entropy
s = gas.entropy_mole           # J/mol/K
s_mass = gas.entropy_mass        # J/kg/K

# Gibbs energy
g = gas.gibbs_mole             # J/mol
g_mass = gas.gibbs_mass         # J/kg

# Internal energy
u = gas.int_energy_mole         # J/mol
u_mass = gas.int_energy_mass     # J/kg

# Heat capacities
cp = gas.cp_mole               # J/mol/K
cp_mass = gas.cp_mass           # J/kg/K
cv = gas.cv_mole               # J/mol/K
cv_mass = gas.cv_mass           # J/kg/K
```

### Partial Molar Properties

```python
# Partial molar enthalpies
h_partial = gas.partial_molar_enthalpies
h_H2 = h_partial[gas.species_index('H2')]

# Partial molar Gibbs energies
g_partial = gas.partial_molar_gibbs
g_H2 = g_partial[gas.species_index('H2')]

# Chemical potentials
mu = gas.chemical_potentials
mu_H2 = mu[gas.species_index('H2')]
```

### Derivatives

```python
# Derivative of enthalpy w.r.t. temperature
dh_dT = gas.dhdT_const_P  # J/mol/K

# Derivative of entropy w.r.t. temperature
ds_dT = gas.dsdT_const_P  # J/mol/K²

# Derivative of pressure w.r.t. temperature
dP_dT = gas.dPdT_const_V  # Pa/K
```

## Species Properties

### Species Information

```python
# Number of species
n_species = gas.n_species

# Species names
species_names = gas.species_names
print(species_names)

# Species molecular weights
mw = gas.molecular_weights
mw_H2 = mw[gas.species_index('H2')]

# Species charges
charges = gas.charges
```

### Composition

```python
# Mole fractions
X = gas.X                    # Array of mole fractions
X_H2 = X[gas.species_index('H2')]

# Mass fractions
Y = gas.Y                    # Array of mass fractions
Y_H2 = Y[gas.species_index('H2')]

# Concentrations
C = gas.concentrations         # kmol/m³
C_H2 = C[gas.species_index('H2')]

# Molar densities
rho_molar = gas.molar_density  # kmol/m³
```

### Setting Composition

```python
# Set mole fractions
gas.X = {'H2': 0.5, 'O2': 0.3, 'N2': 0.2}

# Set mass fractions
gas.Y = {'H2': 0.1, 'O2': 0.8, 'N2': 0.1}

# Set by index
X = gas.X
X[gas.species_index('H2')] = 0.7
gas.X = X
```

## Phase Equilibrium

### Chemical Equilibrium

```python
# Minimize Gibbs free energy
gas = ct.Solution('gri30.yaml', 'gas')
gas.TPX = 2000, 101325, 'H2:2, O2:1'

# Equilibrate at constant T, P
gas.equilibrate('HP', solver='gibbs')

print(f"Equilibrium composition: {gas.X}")
print(f"Equilibrium temperature: {gas.T} K")
```

### Adiabatic Equilibrium

```python
# Constant enthalpy and pressure
gas.equilibrate('HP', solver='gibbs')
```

### Isochoric Equilibrium

```python
# Constant internal energy and density
gas.equilibrate('UV', solver='gibbs')
```

### Solver Options

```python
# Available solvers
solvers = ['gibbs', 'vcs', 'newton']

# Equilibrate with specific solver
gas.equilibrate('HP', solver='newton', max_steps=100, tol=1e-10)
```

## Multi-Phase Equilibrium

### Vapor-Liquid Equilibrium

```python
# Create gas and liquid phases
gas = ct.Solution('gas.yaml', 'gas')
liquid = ct.Solution('liquid.yaml', 'liquid')

# Set initial conditions
gas.TPX = 300, 101325, 'H2O:0.5'
liquid.TPX = 300, 101325, 'H2O:0.5'

# Equilibrate
phases = [gas, liquid]
ct.equilibrate(phases, 'HP', 101325)

print(f"Gas phase: {gas.X}")
print(f"Liquid phase: {liquid.X}")
```

### Condensed Phase Equilibrium

```python
# Solid-gas equilibrium
solid = ct.Solution('solid.yaml', 'solid')
gas = ct.Solution('gas.yaml', 'gas')

phases = [solid, gas]
ct.equilibrate(phases, 'HP', 101325)
```

## Property Calculations

### Adiabatic Flame Temperature

```python
# Calculate adiabatic flame temperature
gas = ct.Solution('gri30.yaml', 'gas')
gas.TPX = 300, 101325, 'H2:2, O2:1'

# Equilibrate at constant enthalpy
gas.equilibrate('HP', solver='gibbs')

T_ad = gas.gas.T
print(f"Adiabatic flame temperature: {T_ad} K")
```

### Heat of Reaction

```python
# Calculate heat of reaction
gas1 = ct.Solution('gri30.yaml', 'gas')
gas1.TPX = 300, 101325, 'H2:2, O2:1'

h_reactants = gas1.enthalpy_mole

# Equilibrate
gas2 = ct.Solution('gri30.yaml', 'gas')
gas2.TPX = 300, 101325, 'H2:2, O2:1'
gas2.equilibrate('HP', solver='gibbs')

h_products = gas2.enthalpy_mole
delta_H = h_products - h_reactants

print(f"Heat of reaction: {delta_H} J/mol")
```

### Specific Heat Ratio

```python
# Calculate gamma = cp/cv
gamma = gas.cp_mole / gas.cv_mole
print(f"Specific heat ratio: {gamma:.4f}")
```

## Common Applications

### Combustion Calculations

```python
# Adiabatic flame temperature
gas = ct.Solution('gri30.yaml', 'gas')
gas.TPX = 300, 101325, 'CH4:1, O2:2')
gas.equilibrate('HP', solver='gibbs')
T_ad = gas.T

# Equilibrium composition
print(f"Products: {gas.X}")
```

### Equilibrium Constants

```python
# Calculate equilibrium constant K
gas = ct.Solution('gri30.yaml', 'gas')
gas.TPX = 1500, 101325, 'H2:0.5, O2:0.5, H2O:0.0'

# Equilibrate
gas.equilibrate('HP', solver='gibbs')

# Calculate K
K = gas.equilibrium_constants
print(f"Equilibrium constants: {K}")
```

### Phase Diagrams

```python
# Calculate phase boundaries
# Vary temperature and equilibrate
temperatures = np.linspace(300, 2000, 100)

for T in temperatures:
    gas.T = T
    gas.equilibrate('HP', solver='gibbs')
    # Record composition
```

## Numerical Considerations

### Convergence

**Equilibrium solver not converging:**
- Try different solver (vcs, newton)
- Increase max_steps
- Relax tolerance
- Check initial guess

### Accuracy

**Improve accuracy:**
- Use high-precision thermodynamic data
- Check thermodynamic consistency
- Verify phase model
- Use appropriate EOS

### Performance

**Speed up calculations:**
- Use compiled mechanisms (CTI)
- Cache equilibrium calculations
- Use appropriate solver
- Consider mechanism reduction

## Troubleshooting

### Invalid State

**Issue:** Cannot set state

**Solutions:**
- Check phase model
- Verify composition sums to 1
- Check temperature/pressure ranges
- Ensure species exist in mechanism

### Equilibrium Failure

**Issue:** Equilibrium solver fails

**Solutions:**
- Try different initial guess
- Use different solver
- Check thermodynamic data
- Verify phase stability

### Property Errors

**Issue:** Invalid property values

**Solutions:**
- Check state is valid
- Verify phase model
- Check units
- Consult documentation
