# Python Automation

OpenFOAM provides powerful Python scripting capabilities for automation and customization.

## Python Scripting

### Basic Script Execution

```python
import subprocess

# Run OpenFOAM command
subprocess.run(['blockMesh', 'cavity'])
```

### Parallel Execution

```python
import subprocess

# Run in parallel with 4 cores
subprocess.run(['mpirun', '-np', '4', 'blockMesh', 'cavity'])
```

### Output Capture

```python
import subprocess

# Capture output
result = subprocess.run(['simpleFoam'], capture_output=True, text=True)
print(result.stdout)
```

## PyFoam API

### Mesh Generation

```python
from PyFoam.RunDictionary.ParsedBlockMeshDict import ParsedBlockMeshDict

# Parse blockMeshDict
args = ParsedParameterFile().parseArgs([
    'blockMeshDict', 'region',
    'convertToMeters', 'mergePatchPairs'
])

# Generate mesh
blockMesh = ParsedBlockMeshDict(args).run()
```

### Field Initialization

```python
from PyFoam.RunDictionary.ParsedSetFieldsDict import ParsedSetFieldsDict

# Parse setFieldsDict
args = ParsedParameterFile().parseArgs([
    'setFieldsDict', 'region',
    'fields', 'U', 'p', 'T'
'  ])

# Initialize fields
setFields = ParsedSetFieldsDict(args).run()
```

### Application Control

```python
from PyFoam.RunDictionary.ParsedApplicationDict import ParsedApplicationDict

# Parse applicationDict
args = ParsedParameterFile().parseArgs([
    'application', 'startFrom', 'stopAt', 'deltaT',
    'writeControl', 'writeFormat', 'writePrecision',
    'solver', 'maxCo', 'maxAlphaCo'
])

# Create application
application = ParsedApplicationDict(args).run()
```

## Custom Solvers

### Creating Custom Solver

```python
from PyFoam.RunDictionary import FoamFileParser, FoamFileGenerator

# Create custom solver
controlDict = {
    'application': {
        'solver': 'myCustomSolver',
        'startFrom': 'latestTime',
        'deltaT': 0.001
    }
}

# Write control dictionary
parser = FoamFileParser(FoamFile('controlDict'))
parser.write(controlDict)
```

### Custom Functions

```python
from PyFoam.basic import FoamFileParser, FoamFileGenerator

# Add custom function
controlDict = {
    'application': {
        'functions': {
            'myCustomFunction': {
                'type': 'coded'
            }
        }
    }
    }
}

parser = FoamFileParser(FoamFile('controlDict'))
parser.write(controlDict)
```

## Post-Processing

### Field Sampling

```python
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile

# Sample field at points
args = ParsedParameterFile().parseArgs([
    'fieldDataDict', 'region',
    'startTime', 'endTime',
    'sampleMode': 'nearestCell'
])

fieldData = ParsedFieldDataDict(args).run()
```

### Time Series Extraction

```python
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile

# Extract time series
args = ParsedParameterFile().parseArgs([
    'sampleDict', 'region',
    'startTime', 'endTime'
])

sampleDict = ParsedSampleDict(args).run()
```

### Force Coefficients

```python
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile

# Calculate forces
args = ParsedParameterFile().parseArgs([
    'forcesDict', 'region',
    'startTime', 'endTime'
])

forcesDict = ParsedForcesDict(args).run()
```

## Parametric Studies

### Parametric Sweep

```python
import numpy as np
from PyFoam.RunDictionary import FoamFileParser

# Parametric sweep
velocities = np.linspace(0.1, 2.0, 10)
results = []

for vel in velocities:
    # Modify controlDict
    controlDict['application']['U']['value'] = [vel, 0]
    
    # Write and run
    parser = FoamFileParser(FoamFile('controlDict'))
    parser.write(controlDict)
    
    subprocess.run(['simpleFoam'])
    
    # Collect results
    results.append(vel)
```

### Design of Experiments

```python
import numpy as np
from PyFoam.RunDictionary import FoamFileParser

# Factorial design
factors = [0.5, 1.0, 1.5, 2.0]
velocities = [1.0]

for factor in factors:
    for vel in velocities:
        # Run simulation
        controlDict['application']['U']['value'] = [vel * factor, 0]
        
        parser = FoamFileParser(FoamFile('controlDict'))
        parser.write(controlDict)
        
        subprocess.run(['simpleFoam'])
```

## Batch Processing

### Multiple Cases

```python
import subprocess

# Run multiple cases
cases = ['cavity', 'damBreak', 'hotRoom']

for case in cases:
    subprocess.run(['blockMeshDict', case])
    subprocess.run(['simpleFoam'])
```

### Cluster Execution

```python
import subprocess

# Run on cluster
subprocess.run([
    'mpirun', '-np', '16', '-npernode', '4',
    'blockMeshDict', 'cavity'
])
```

## Advanced Automation

### Custom Mesh Generation

```python
from PyFoam.RunDictionary import FoamFileParser

# Custom mesh generation
controlDict = {
    'application': {
        'functions': {
            'generateMesh': {
                'type': 'coded'
            }
        }
    }
    }
}

parser = FoamFileParser(FoamFile('controlDict'))
parser.write(controlDict)
```

### Dynamic Boundary Conditions

```python
import math

# Time-dependent boundary
def get_inlet_velocity(t):
    return 1.0 + 0.5 * math.sin(2 * math.pi * t)

# Update boundary during simulation
# Use in custom function or post-processing
```

### Error Handling

```python
import subprocess

try:
    subprocess.run(['simpleFoam'])
except subprocess.CalledProcessError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"Exception: {e}")
```

## Performance Optimization

### Vectorized Operations

```python
import numpy as np

# Vectorized field operations
# Use numpy arrays for efficient calculations
# Avoid Python loops for large arrays
```

### Memory Management

```python
# Use appropriate data structures
# Delete unnecessary variables
# Use generators instead of lists
```

### Caching Results

```python
# Cache expensive calculations
# Use memoization for repeated operations
# Store intermediate results
```

## Common Issues and Solutions

### Python Path Issues

**Problem**: Python cannot find OpenFOAM modules

**Solutions**:
- Source OpenFOAM environment
- Add OpenFOAM to PYTHONPATH
- Use foamInstallation script

### Import Errors

**Problem**: Cannot import PyFoam modules

**Solutions**:
- Install PyFoam package
- Check Python version compatibility
- Use correct import paths

### File Permission Issues

**Problem**: Cannot write to case files

**Solutions**:
- Check directory permissions
- Run as appropriate user
- Use sudo if necessary

### Convergence Issues

**Problem**: Python script doesn't converge

**Solutions**:
- Check OpenFOAM convergence criteria
- Verify boundary conditions
- Check time step size
- Monitor residuals

### Performance Issues

**Problem**: Python automation is slow

**Solutions**:
- Use vectorized operations
- Minimize file I/O
- Use compiled solvers when possible
- Batch operations efficiently

## Best Practices

1. **Use PyFoam API**: Prefer over direct file manipulation
2. **Handle errors**: Always include error handling
3. **Validate inputs**: Check parameters before execution
4. **Use subprocess**: For running OpenFOAM commands
5. **Document scripts**: Add comments for clarity
6. **Test locally**: Test before batch execution
7. **Use appropriate data types**: Match OpenFOAM expectations
8. **Clean up**: Remove temporary files after completion
