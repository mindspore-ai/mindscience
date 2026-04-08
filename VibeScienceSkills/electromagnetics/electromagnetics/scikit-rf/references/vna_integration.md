# VNA Integration in scikit-rf

Virtual Network Analyzer (VNA) integration for automated measurements.

## VNA Overview

VNA integration provides automated control and data acquisition from vector network analyzers.

## Supported VNAs

### Keysight PNA

```python
from skrf.vi.vna.keysight import PNA

# Connect to PNA
instr = PNA(address="TCPIP0::10.0.0.1::INSTR")

# Set frequency
freq = rf.Frequency(1, 10, 101, 'GHz')
instr.frequency = freq

# Get S-parameter network
ntwk = instr.get_snp_network(ports=(1, 2))

# Plot results
ntwk.s21.plot_s_db()
plt.show()
```

### HP 8510C PNA

```python
from skrf.vi.vna.hp import HP8510C

# Connect to HP VNA
instr = HP8510C(address="TCPIP0::10.0.0.1::INSTR")

# Set frequency
freq = rf.Frequency(1, 10, 101, 'GHz')
instr.frequency = freq

# Get S-parameter network
ntwk = instr.get_snp_network(ports=(1, 2))

# Plot results
ntwk.s21.plot_s_db()
plt.show()
```

## VNA Control

### Frequency Control

```python
# Set frequency range
freq = rf.Frequency(1, 10, 101, 'GHz')
instr.frequency = freq

# Set single frequency
instr.frequency = 2.5e9  # 2.5 GHz

# Get current frequency
current_freq = instr.frequency
print(f"Current frequency: {current_freq}")
```

### Measurement Control

```python
# Get S-parameter network
ntwk = instr.get_snp_network(ports=(1, 2))

# Get S-parameter network with averaging
ntwk = instr.get_snp_network(ports=(1, 2), averages=10)

# Get S-parameter network with sweep
ntwk = instr.get_snp_network(ports=(1, 2), sweep_points=201)
```

### Switch Term Measurement

```python
# Get switch terms
switch_terms = instr.get_switch_terms()

# Switch terms account for internal switching
# in the VNA error model
print(f"Switch terms: {switch_terms}")
```

## Measurement Automation

### Automated Measurement Script

```python
import skrf as rf
from skrf.vi.vna.keysight import PNA
import matplotlib.pyplot as plt

# Connect to VNA
instr = PNA(address="TCPIP0::10.0.0.1::INSTR")

# Set frequency
freq = rf.Frequency(1, 10, 101, 'GHz')
instr.frequency = freq

# Measure DUT
ntwk = instr.get_snp_network(ports=(1, 2))

# Save measurement
ntwk.write_touchstone('measurements/dut.s2p')

# Plot results
rf.stylely()
ntwk.s21.plot_s_db()
plt.title('DUT Measurement')
plt.show()
```

### Batch Measurement

```python
# Measure multiple devices
devices = ['device1', 'device2', 'device3']

for device in devices:
    # Connect device
    instr = PNA(address=f"TCPIP0::10.0.0.1::{device}")
    
    # Set frequency
    instr.frequency = freq
    
    # Measure
    ntwk = instr.get_snp_network(ports=(1, 2))
    
    # Save
    ntwk.write_touchstone(f'measurements/{device}.s2p')
    print(f"Measured {device}")
```

## VNA Calibration

### Switch Term Calibration

```python
# Measure switch terms
from skrf.vi.vna.keysight import PNA

instr = PNA(address="TCPIP0::10.0.0.1::INSTR")
switch_terms = instr.get_switch_terms()

# Use switch terms in calibration
from skrf.calibration import EightTerm

cal = EightTerm(
    ideals=ideals,
    measured=measured,
    switch_terms=switch_terms
)

cal.run()
```

### VNA Self-Calibration

```python
# Perform VNA self-calibration
instr.calibrate()

# Verify calibration
print("Calibration complete")
```

## Error Handling

### Connection Errors

```python
try:
    instr = PNA(address="TCPIP0::10.0.0.1::INSTR")
except Exception as e:
    print(f"Connection failed: {e}")
    # Handle connection error
```

### Measurement Errors

```python
try:
    ntwk = instr.get_snp_network(ports=(1, 2))
except Exception as e:
    print(f"Measurement failed: {e}")
    # Handle measurement error
```

## Best Practices

### VNA Setup

1. **Use stable connections**: Ensure network connection is stable
2. **Set appropriate frequency**: Match VNA capabilities
3. **Use averaging**: Reduce measurement noise
4. **Include switch terms**: Essential for accurate calibration

### Measurement Strategy

1. **Calibrate regularly**: VNA calibration drifts over time
2. **Use appropriate averages**: Balance speed and accuracy
1. **Verify measurements**: Check for obvious errors
4. **Document setup**: Record VNA configuration

### Error Handling

1. **Handle connection errors**: Implement retry logic
2. **Validate measurements**: Check for invalid values
3. **Log errors**: Record errors for debugging
4. **Provide feedback**: Inform user of issues

## Troubleshooting

### Connection Issues

**Problem**: Cannot connect to VNA

**Solutions**:
1. Check network connection
2. Verify VNA address
3. Check VNA is powered on
4. Verify VNA is not in use by another application

### Measurement Failures

**Problem**: Measurements fail or return invalid data

**Solutions**:
1. Check VNA calibration
2. Verify frequency range
3. Check port configuration
4. Try different measurement parameters

### Switch Term Issues

**Problem**: Switch terms measurement fails

**Solutions**:
1. Check VNA supports switch term measurement
2. Verify measurement setup
3. Use alternative calibration method
4. Contact VNA manufacturer

### Performance Issues

**Problem**: Measurements are slow

**Solutions**:
1. Reduce number of frequency points
2. Use fewer averages
3. Check network latency
4. Optimize measurement script