# Calibration in scikit-rf

Calibration corrects systematic errors in VNA measurements using ideal responses.

## Calibration Overview

Calibration removes systematic errors from VNA measurements by comparing measured responses to known ideal responses. scikit-rf supports multiple calibration algorithms for different applications.

## One-Port Calibration

### Basic One-Port Calibration

```python
import skrf as rf
from skrf.calibration import OnePort

# Load ideal and measured responses
ideals = [
    rf.Network('ideal/short.s1p'),
    rf.Network('ideal/open.s1p'),
    rf.Network('ideal/load.s1p')
]

measured = [
    rf.Network('measured/short.s1p'),
    rf.Network('measured/open.s1p'),
    rf.Network('measured/load.s1p')
]

# Create calibration
cal = OnePort(ideals=ideals, measured=measured)

# Run calibration
cal.run()

# Apply to DUT
dut = rf.Network('data/dut.s1p')
dut_calibrated = cal.apply_cal(dut)
```

### Calibration Parameters

```python
# Create calibration with parameters
cal = OnePort(
    ideals=ideals,
    measured=measured,
    name='my_calibration',  # optional name
    family='eight_term',  # error model family
    deg=1,  # polynomial degree
    n_thrus=100,  # number of coefficients
    is_reciprocal=False  # if network is reciprocal
)
```

### Self-Calibration (TRL)

```python
from skrf.calibration import TRL

# TRL calibration (no ideals required)
cal_trl = TRL(measured=measured)

# Run and apply
cal_trl.run()
dut_calibrated = cal_trl.apply_cal(dut)
```

## Two-Port Calibration

### SOLT Calibration

```python
from skrf.calibration import SOLT

# Load two-port networks
ideals = [
    rf.Network('ideal/short.s2p'),
    rf.Network('ideal/open.s2p'),
    rf.Network('ideal/load.s2p'),
    rf.Network('ideal/thru.s2p')
]

measured = [
    rf.Network('measured/short.s2p'),
    rf.Network('measured/open.s2p'),
    rf.Network('measured/load.s2p'),
    rf.Network('measured/thru.s2p')
]

# Create SOLT calibration
cal_solt = SOLT(ideals=ideals, measured=measured)

# Run and apply
cal_solt.run()
dut_calibrated = cal_solt.apply_cal(dut)
```

### EightTerm Calibration

```python
from skrf.calibration import EightTerm

# EightTerm calibration
cal_8term = EightTerm(
    ideals=ideals,
    measured=measured,
    switch_terms=switch_terms  # optional switch terms
)

# Run and apply
cal_8term.run()
dut_calibrated = cal_8term.apply_cal(dut)
```

### Switch Terms

Switch terms account for internal VNA switching behavior:

```python
# Measure switch terms
from skrf.vi.vna.keysight import PNA

instr = PNA(address="TCPIP0::10.0.0.1::INSTR")
switch_terms = instr.get_switch_terms()

# Use in calibration
cal_8term = EightTerm(
    ideals=ideals,
    measured=measured,
    switch_terms=switch_terms
)
```

### Isolation Calibration

```python
# Create isolation network
isolation = rf.Network('data/isolation.s2p')

# Use in calibration
cal_solt = SOLT(
    ideals=ideals,
    measured=measured,
    isolation=isolation  # optional isolation calibration
)
```

## Advanced Calibration

### SixteenTerm Calibration

```python
from skrf.calibration import SixteenTerm

# SixteenTerm calibration (better leakage correction)
cal_16term = SixteenTerm(
    ideals=ideals,
    measured=measured
)

# Run and apply
cal_16term.run()
dut_calibrated = cal_16term.apply_cal(dut)
```

### TwelveTerm Calibration

```python
from skrf.calibration import TwelveTerm

# TwelveTerm calibration
cal_12term = TwelveTerm(
    ideals=ideals,
    measured=measured
)

# Run and apply
cal_12term.run()
dut_calibrated = cal_12term.apply_cal(dut)
```

### Using One-Port Ideals in Two-Port Calibration

```python
# Convert one-port ideals to two-port
short_1p = rf.Network('ideal/short.s1p')
short_2p = rf.network.two_port_reflect(short_1p, short_1p)

# Use in two-port calibration
ideals = [short_2p, ...]
```

## Calibration Quality

### Residual Analysis

```python
# Get calibration residuals
residuals = cal.residuals

# Plot residuals
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
residuals.s11.plot_s_db(m=1, n=0, label='S11 residual')
residuals.s21.plot_s_db(m=1, n=0, label='S21 residual')
plt.legend()
plt.title('Calibration Residuals')
plt.show()
```

### Error Coefficients

```python
# Get error coefficients
coefficients = cal.coef

# Print coefficients
print(f"Error coefficients: {coefficients}")
```

### Calibration Uncertainty

```python
# Get calibration uncertainty
uncertainty = cal.uncertainty

# Plot uncertainty
uncertainty.plot_s_db(m=1, n=0, label='Uncertainty')
plt.legend()
plt.title('Calibration Uncertainty')
plt.show()
```

## Saving and Loading Calibration

### Save Calibration

```python
# Save calibration to file
cal.write('data/my_calibration.cal')

# Or using general I/O
rf.io.write('data/my_calibration.cal', cal)
```

### Load Calibration

```python
# Load calibration from file
cal = rf.io.read('data/my_calibration.cal')

# Apply loaded calibration
dut_calibrated = cal.apply_cal(dut)
```

## Calibration Workflow

### Complete Calibration Workflow

```python
import skrf as rf
from skrf.calibration import OnePort
import matplotlib.pyplot as plt

# 1. Load ideal and measured responses
ideals = rf.io.read_all('ideals/', contains='s1p')
measured = rf.io.read_all('measured/', contains='s1p')

# 2. Create calibration
cal = OnePort(
    ideals=[ideals[k] for k in ['short', 'open', 'load']],
    measured=[measured[k] for k in ['short', 'open', 'load']],
    name='SOL_calibration'
)

# 3. Run calibration
print("Running calibration...")
cal.run()

# 4. Analyze calibration quality
print("Analyzing calibration quality...")
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
cal.residuals.s11.plot_s_db(m=1, n=0)
plt.title('S11 Residuals')

plt.subplot(2, 2, 2)
cal.residuals.s21.plot_s_db(m=1, n=0)
plt.title('S21 Residuals')

plt.subplot(2, 2, 3)
cal.uncertainty.plot_s_db(m=1, n=0)
plt.title('Uncertainty')

plt.subplot(2, 2, 4)
cal.coef.plot_s_db(m=1, n=0)
plt.title('Error Coefficients')

plt.tight_layout()
plt.show()

# 5. Apply calibration to DUTs
print("Applying calibration to DUTs...")
duts = rf.io.read_all('duts/', contains='s1p')

for name, dut in duts.items():
    dut_calibrated = cal.apply_cal(dut)
    dut_calibrated.name = f"{name}_calibrated"
    dut_calibrated.write_touchstone(f'calibrated/{name}.s1p')
    print(f"  Calibrated: {name}")

# 6. Save calibration
cal.write('data/my_calibration.cal')
print("Calibration saved to data/my_calibration.cal")
```

## Best Practices

### Standard Selection

1. **Match DUT characteristics**: Use standards similar to DUT
2. **Cover frequency range**: Ensure standards cover DUT frequency range
3. **Include multiple states**: Short, open, load, thru
4. **Use appropriate standards**: Match connector types and impedances

### Calibration Algorithm

1. **One-port**: Simple measurements, low accuracy requirements
2. **Two-port**: High accuracy, switch-term correction
3. **TRL**: Self-calibration, no ideals required
4. **EightTerm**: Best general-purpose two-port calibration

### Switch Terms

1. **Always include switch terms**: Essential for accurate calibration
2. **Measure properly**: Use low-loss thru standard
3. **Update regularly**: Switch terms can change over time

### Quality Verification

1. **Check residuals**: Should be small across frequency range
2. **Verify uncertainty**: Should be within acceptable limits
3. **Test with known devices**: Verify calibration works correctly

## Troubleshooting

### Poor Calibration Quality

**Problem**: Large residuals or high uncertainty

**Solutions**:
1. Verify ideal responses are correct
2. Check measurement quality
3. Include switch terms
4. Try different calibration algorithm
5. Increase number of standards

### Frequency Mismatch

**Problem**: Calibration fails due to frequency mismatch

**Solutions**:
1. Resample networks to common frequencies
2. Check frequency ranges overlap
3. Verify frequency units are consistent

### Port Count Issues

**Problem**: Network port count incompatibility

**Solutions**:
1. Check all networks have same port count
2. Use `two_port_reflect` for one-port ideals
3. Verify network creation parameters

### Calibration Application Fails

**Problem**: `apply_cal` fails or produces invalid results

**Solutions**:
1. Verify DUT has compatible frequency range
2. Check DUT port count matches calibration
3. Verify calibration was run successfully
4. Check for NaN or Inf values in DUT