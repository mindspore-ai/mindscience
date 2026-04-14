# CoolProp Interfaces

CoolProp provides multiple programming language interfaces for different applications.

## Python Interface

### Installation

```bash
pip install CoolProp
```

### Basic Usage

```python
from CoolProp.CoolProp import PropsSI

# Calculate density
rho = PropsSI('D', 'T', 300.0, 'P', 101325.0, 'Water')
print(f"Density: {rho} kg/m³")
```

### High-Level Interface

```python
from CoolProp.CoolProp import CoolProp

# Create CoolProp instance
CP = CoolProp.CoolProp()

# Set configuration
CP.set_config_bool(CP.OVERWRITE_FLUIDS, True)

# Calculate properties
rho = CP.PropsSI('D', 'T', 300.0, 'P', 101325.0, 'Water')
```

### Humid Air Interface

```python
from CoolProp.CoolProp import HumidAir

# Create humid air instance
ha = HumidAir(T=298.15, P=101325.0, RH=0.5)

# Get properties
W_bulb = ha.W_bulb()
T_dp = ha.T_dp()
```

## C++ Interface

### Static Library

```cpp
#include "CoolProp.h"

int main() {
    double rho = PropsSI("D", "T", 300.0, "P", 101325.0, "Water");
    std::cout << "Density: " << rho << " kg/m³" << std::endl;
    return 0;
}
```

### Shared Library

```cpp
#include "CoolProp.h"

int main() {
    CoolProp::CoolProp CP;
    double rho = CP.PropsSI("D", "T", 300.0, "P", 101325.0, "Water");
    std::cout << "Density: " << rho << " kg/m³" << std::endl;
    return 0;
}
```

## MATLAB Interface

### Basic Usage

```matlab
% Calculate water density
rho = PropsSI('D', 'T', 300, 'P', 101325, 'Water');
disp(['Density:', rho, 'kg/m³']);
```

### Vector Calculations

```matlab
% Vector inputs
T = [280, 290, 300, 310];
P = [101325, 101325, 101325, 101325];

% Vector outputs
outputs = {'D', 'Cp', 'viscosity'};
rho = PropsSImulti(outputs, 'T', T, 'P', P, '', 'Water');
```

## Excel Interface

### Basic Usage

```vba
' Calculate water density in Excel
=PropsSI("D", "T", 300, "P", 101325, "Water")
```

### Multiple Calculations

```vba
' Calculate multiple properties
=PropsSI("D", "T", 300, "P", 101325, "Water")
=Props
```

## Other Interfaces

### Modelica

```modelica
// Modelica interface
rho = CoolProp.PropsSI("D", "T", 300, "P", 101325, "Water");
```

### Octave

```octave
% Octave interface
rho = PropsSI('D', 'T', 300.0, 'P', 101325.0, 'Water');
```

### Java

```java
// Java interface
CoolProp CP = new CoolProp();
double rho = CP.PropsSI("D", "T", 300.0, "P", 101325.0, "Water");
```

## Interface Selection Guide

| Application | Recommended Interface | Reason |
|------------|---------------------|---------|
| Python scripts | Python | Most flexible, easiest |
| Scientific computing | Python/C++ | High performance |
| Data analysis | MATLAB/Excel | Easy integration |
| Industrial applications | C++ | Fastest execution |
| Web applications | Python/JavaScript | Web integration |
| Control systems | C#/VB.NET | Industrial control |

## Performance Considerations

### Python Performance

| Operation | Speed | Notes |
|-----------|-------|-------|
| Single property | Fast | Direct C++ call |
| Multiple properties | Fast | Vectorized operations |
| Batch calculations | Medium | Python overhead |
| Humid air calculations | Fast | Optimized code |

### C++ Performance

| Operation | Speed | Notes |
|-----------|-------|-------|
| Single property | Fastest | Native C++ |
| Vector operations | Fastest | Optimized loops |
| Multiple fluids | Fast | Efficient memory use |
| Complex mixtures | Fast | Native performance |

### MATLAB Performance

| Operation | Speed | Notes |
|-----------|-------|-------|
| Single property | Medium | MATLAB overhead |
| Vector operations | Medium | MATLAB vectorization |
| Multiple properties | Slow | Multiple calls |
| Batch calculations | Slow | High overhead |

## Common Issues and Solutions

### Installation Issues

**Problem**: Cannot import CoolProp

**Solutions**:
- Verify installation: `pip install CoolProp`
- Check Python version compatibility
- Try reinstalling: `pip install --force-reinstall CoolProp`
- Check system PATH

### Calculation Failures

**Problem**: Property calculation fails

**Solutions**:
- Check input pair validity
- Verify fluid name spelling
- Check phase specification
- Ensure sufficient input parameters
- Try different input pair

### Accuracy Issues

**Problem**: Results seem inaccurate

**Solutions**:
- Use high-accuracy backend (IF97)
- Check reference state settings
- Verify input pair validity
- Compare with experimental data
- Check critical properties

### Performance Issues

**Problem**: Calculations too slow

**Solutions**:
- Use appropriate input pairs: ('T', 'P') fastest
- Use vectorized operations
- Batch similar calculations
- Use C++ interface for performance
- Use tabular interpolation for speed

## Best Practices

1. **Choose appropriate interface**: Python for flexibility, C++ for speed
2. **Use vectorized operations**: For multiple calculations
3. **Handle exceptions**: Catch and handle calculation failures
4. **Validate inputs**: Check input ranges before calculations
5. **Use appropriate backend**: HEOS, IF97, or REFPROP
6. **Batch calculations**: Group similar operations for efficiency
7. **Cache results**: For repeated calculations
8. **Verify results**: Check physical reasonableness
