---
name: scikit-rf
description: Open-source Python package for RF/Microwave engineering. Use for:(1) Network analysis and simulation (S-parameters, Z-parameters, ABCD, T-parameters), (2) VNA calibration and error correction (OnePort, TwoPort, SOLT, EightTerm), (3) Vector network analyzer (VNA) control and data acquisition, (4) Transmission line media modeling (CPW, coax, waveguide), (5) Network set statistical analysis and uncertainty bounds, (6) Quality factor calculations and resonator analysis, (7) Circuit and system-level RF design, (8) Touchstone file I/O for network parameter data.
---

# scikit-rf: RF/Microwave Engineering Toolkit

scikit-rf (skrf) is an open-source, BSD-licensed Python package for RF/microwave engineering providing a modern, object-oriented library for network analysis, calibration, and measurement automation.

## Quick Start

Basic network analysis workflow:

```python
import skrf as rf

# Load network from Touchstone file
ring_slot = rf.Network('data/ring_slot.s2p')

# Plot S-parameters
ring_slot.plot_s_db()

# Access network properties
print(f"Frequency range: {ring_slot.frequency}")
print(f"Port impedance: {ring_slot.z0}")
```

## Core Concepts

### Network Objects

The central object in skrf is a **Network** representing an N-port microwave network:

```python
# Create network from Touchstone file
network = rf.Network('data/device.s2p')

# Create network from S-parameters
freq = rf.Frequency(1, 10, 101, 'GHz')
s = np.random.uniform(size=(101, 2, 2)) + 1j*np.random.uniform(size=(101, 2, 2))
network = rf.Network(frequency=freq, s=s, name='my_network')
```

### Network Parameters

Networks support multiple parameter representations:

- **S-parameters**: Scattering parameters (default)
- **Z-parameters**: Impedance parameters
- **Y-parameters**: Admittance parameters
- **ABCD-parameters**: ABCD matrix
- **T-parameters**: Transfer scattering parameters
- **H-parameters**: Hybrid parameters

### Frequency Objects

Frequency arrays are managed through **Frequency** objects:

```python
# Create frequency range
freq = rf.Frequency(1, 10, 101, 'GHz')

# Access frequency values
print(freq.f)  # numpy array of frequencies
print(freq.f_scaled)  # frequencies scaled to 1.0 GHz

# Slice by frequency
network['1-5ghz']  # network subset from 1-5 GHz
```

### Port Impedance

Characteristic impedance is stored in the **z0** property:

```python
# Access port impedance
print(network.z0)  # numpy array of shape (nfreq, nports, nports)

# Create network with specific impedance
network = rf.Network(frequency=freq, s=s, z0=50)  # 50 Ohm for all ports
```

## Common Workflows

### Workflow 1: Network Analysis

Load and analyze microwave networks:

```python
import skrf as rf
import matplotlib.pyplot as plt

# Load network
network = rf.Network('data/filter.s2p')

# Plot S-parameters
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
network.s11.plot_s_db(m=1, n=0, label='S11')
plt.legend()

plt.subplot(2, 2, 2)
network.s21.plot_s_db(m=1, n=0, label='S21')
plt.legend()

plt.subplot(2, 2, 3)
network.s12.plot_s_db(m=1, n=0, label='S12')
plt.legend()

plt.subplot(2, 2, 4)
network.s22.plot_s_db(m=1, n=0, label='S22')
plt.legend()

plt.tight_layout()
plt.show()
```

### Workflow 2: Network Cascading

Cascade and de-embed networks:

```python
import skrf as rf

# Load networks
line = rf.Network('data/line.s2p')
short = rf.Network('data/short.s2p')

# Cascade networks
cascaded = line ** short

# De-embed network
deembedded = line.inv ** short

# Plot results
cascaded.s21.plot_s_db(label='Cascaded')
deembedded.s21.plot_s_db(label='De-embedded')
plt.legend()
plt.show()
```

### Workflow 3: VNA Calibration

Calibrate measurements using ideal responses:

```python
import skrf as rf
from skrf.calibration import OnePort

# Load ideal and measured responses
ideals = rf.io.read_all('ideals/', contains='s1p')
measured = rf.io.read_all('measured/', contains='s1p')

# Create calibration
cal = OnePort(
    ideals=[ideals[k] for k in ['short', 'open', 'load']],
    measured=[measured[k] for k in ['short', 'open', 'load']]
)

# Run calibration
cal.run()

# Apply to DUT
dut = rf.Network('data/dut.s1p')
dut_calibrated = cal.apply_cal(dut)
dut_calibrated.write_touchstone('data/dut_calibrated.s1p')
```

### Workflow 4: Network Set Analysis

Analyze statistical properties of multiple networks:

```python
import skrf as rf
from skrf.networkSet import NetworkSet

# Load multiple networks
networks = rf.io.read_all('data/', contains='measurement')

# Create network set
ns = NetworkSet(networks, name='measurement set')

# Calculate statistics
mean_network = ns.mean_s
std_network = ns.std_s

# Plot uncertainty bounds
ns.plot_uncertainty_bounds_s_db(m=1, n=0, label='S11')
plt.legend()
plt.show()
```

### Workflow 5: Transmission Line Media

Create transmission line models:

```python
import skrf as rf
from skrf.media import CPW, Coaxial

# Create CPW
freq = rf.Frequency(75, 110, 101, 'GHz')
cpw = CPW(freq, w=10e-6, s=5e-6, ep_r=10.6)

# Create transmission line
line = cpw.line(d=90, unit='deg', name='90deg_line')

# Plot characteristic impedance
line.plot_z_re(m=1, n=0)
plt.show()
```

## Network Operations

### Arithmetic Operations

Element-wise operations on networks:

```python
# Load networks
network1 = rf.Network('data/network1.s2p')
network2 = rf.Network('data/network2.s2p')

# Addition
sum_network = network1 + network2

# Subtraction
diff_network = network1 - network2

# Multiplication
prod_network = network1 * network2

# Division
ratio_network = network1 / network2
```

### Network Comparison

Compare networks:

```python
# Equality check
if network1 == network2:
    print("Networks are equal")

# Inequality check
if network1 != network2:
    print("Networks are different")
```

### Network Slicing

Slice networks by frequency or parameter:

```python
# Slice by frequency
network_subset = network['1-5ghz']

# Slice S-parameters
s11_subset = network.s11[:10]  # first 10 frequency points

# Slice by port
s11_port0 = network.s[:, 0, :]  # all parameters for port 0
```

## Network Connections

### Port Connection

Connect specific ports between networks:

```python
# Load networks
tee = rf.Network('data/tee.s2p')
load = rf.Network('data/load.s2p')

# Connect port 1 of tee to port 0 of load
connected = rf.network.connect(tee, 1, load, 0)

# Plot result
connected.s21.plot_s_db()
plt.show()
```

### Multi-port Cascading

Cascade multi-port networks:

```python
# Load 4-port networks
network1 = rf.Network('data/network1.s4p')
network2 = rf.Network('data/network2.s4p')
network3 = rf.Network('data/network3.s4p')

# Cascade networks
result = network1 ** network2 ** network3

# Port mapping:
# network1    network2    network3
# +----+    +----+    +----+
# 0-|1  2|--|0  2|--|0  2|-2
# 1-|3  4|--|1  3|--|1  3|-3
# +----+    +----+    +----+
```

## Calibration Methods

### One-Port Calibration

Simple one-port calibration:

```python
from skrf.calibration import OnePort

# Create calibration
cal = OnePort(
    ideals=[ideal_short, ideal_open, ideal_load],
    measured=[measured_short, measured_open, measured_load]
)

# Run and apply
cal.run()
calibrated_dut = cal.apply_cal(dut)
```

### Two-Port Calibration

Advanced two-port calibration:

```python
from skrf.calibration import SOLT, EightTerm

# SOLT calibration
cal_solt = SOLT(
    ideals=[ideal_short, ideal_open, ideal_load, ideal_thru],
    measured=[measured_short, measured_open, measured_load, measured_thru],
    isolation=isolation_network  # optional
)

# EightTerm calibration
cal_8term = EightTerm(
    ideals=[ideal_short, ideal_open, ideal_load],
    measured=[measured_short, measured_open, measured_load]
)

# Run and apply
cal_solt.run()
cal_8term.run()
```

### Self-Calibration

TRL (self-calibration) method:

```python
from skrf.calibration import TRL

# TRL calibration (no ideals required)
cal_trl = TRL(measured=[measured_short, measured_open, measured_load])

# Run and apply
cal_trl.run()
calibrated_dut = cal_trl.apply_cal(dut)
```

## VNA Integration

### VNA Control

Control vector network analyzers:

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

### Switch Term Measurement

Measure switch terms for calibration:

```python
# Get switch terms
switch_terms = instr.get_switch_terms()

# Switch terms account for internal switching
# in the VNA error model
```

## Media and Transmission Lines

### CPW Media

Coplanar waveguide media:

```python
from skrf.media import CPW

freq = rf.Frequency(75, 110, 101, 'GHz')
cpw = CPW(freq, w=10e-6, s=5e-6, ep_r=10.6)

# Access properties
print(f"Propagation constant: {cpw.gamma}")
print(f"Characteristic impedance: {cpw.z0}")
```

### Coaxial Media

Coaxial transmission line:

```python
from skrf.media import Coaxial

freq = rf.Frequency(1, 10, 101, 'GHz')
coax = Coaxial(frequency=freq, Dint=1e-3, Dout=2e-3)

# Create line
line = coax.line(d=90, unit='deg', name='coax_line')
```

### Rectangular Waveguide

Rectangular waveguide media:

```python
from skrf.media import RectangularWaveguide

freq = rf.Frequency(10, 20, 101, 'GHz')
rwg = RectangularWaveguide(freq, a=2.54e-3, b=1.27e-3)

# Create line
line = rwg.line(d=100, unit='mm', name='rwg_line')
```

## Quality Factor Analysis

### Q-Factor Calculation

Calculate quality factors:

```python
from skrf.qfactor import Qfactor

# Load network
network = rf.Network('data/resonator.s2p')

# Calculate Q-factors
qf = Qfactor(network)

# Get Q-factors
print(f"Unloaded Q: {qf.q_unloaded}")
print(f"Loaded Q: {qf.q_loaded}")
print(f"External Q: {qf.q_external}")
```

### Resonator Analysis

Analyze resonator characteristics:

```python
# Calculate resonator parameters
resonance = qf.resonance

print(f"Resonance frequency: {resonance.f0}")
print(f"Bandwidth: {resonance.bw}")
print(f"Coupling: {resonance.k}")
```

## Plotting and Visualization

### Network Plotting

Plot network parameters:

```python
# Apply skrf plotting style
rf.stylely()

# Plot S-parameters
network.plot_s_db(m=1, n=0, label='S11')
network.plot_s_deg(m=1, n=0, label='S11 phase')
network.plot_s_smith(m=1, n=0)

# Plot Z-parameters
network.plot_z_re(m=1, n=0)
network.plot_z_im(m=1, n=0)

# Plot Y-parameters
network.plot_y_db(m=1, n=0)
```

### Smith Chart Plots

Plot on Smith chart:

```python
# Plot S11 on Smith chart
network.s11.plot_s_smith()

# Plot S21 on Smith chart
network.s21.plot_s_smith()

# Plot S12 on Smith chart
network.s12.plot_s_smith()
```

### Custom Plotting

Create custom plots:

```python
import matplotlib.pyplot as plt

# Custom S-parameter plot
plt.figure(figsize=(10, 6))
network.s11.plot_s_db(m=1, n=0, label='|S11|')
network.s21.plot_s_db(m=1, n=0, label='|S21|')
plt.xlabel('Frequency (GHz)')
plt.ylabel('Magnitude (dB)')
plt.title('S-parameters')
plt.legend()
plt.grid(True)
plt.show()
```

## File I/O

### Reading Networks

Read networks from files:

```python
# Read single Touchstone file
network = rf.Network('data/device.s2p')

# Read all skrf files in directory
networks = rf.io.read_all('data/', contains='s2p')

# Read specific files
files = ['data/network1.s2p', 'data/network2.s2p']
networks = rf.io.read_all(files=files)
```

### Writing Networks

Write networks to files:

```python
# Write Touchstone file
network.write_touchstone('data/output.s2p')

# Write pickle file
rf.io.write('data/output.ntwk', network)

# Write all networks
for name, network in networks.items():
    rf.io.write(f'data/{name}.ntwk', network)
```

## Best Practices

### Network Organization

1. **Use descriptive names**: `network = rf.Network('data/lowpass_filter.s2p')`
2. **Organize by function**: `filters/`, `amplifiers/`, `calibrations/`
3. **Version control**: Track Touchstone files in git

### Calibration Strategy

1. **Use appropriate standards**: Match standards to DUT characteristics
2. **Include switch terms**: Essential for accurate calibration
3. **Verify calibration**: Check calibrated response against ideals

### Frequency Management

1. **Use consistent units**: Standardize on GHz or Hz
2. **Adequate frequency points**: Balance resolution and speed
3. **Check frequency alignment**: Ensure networks have compatible frequencies

### Data Validation

1. **Check network compatibility**: Verify port counts and frequency ranges
2. **Validate measurements**: Check for obvious errors or outliers
3. **Document assumptions**: Record measurement conditions and setup

## Troubleshooting

### Import Errors

**Problem**: `ImportError: No module named 'skrf'`

**Solution**: Install scikit-rf
```bash
pip install scikit-rf
```

### Frequency Mismatch

**Problem**: `IndexError: Networks must have same frequency`

**Solution**: Resample or interpolate networks
```python
network1.resample(len(network2.frequency.f))
```

### Port Impedance Issues

**Problem**: Unexpected impedance values

**Solution**: Check z0 parameter
```python
print(network.z0)  # inspect impedance values
```

### Calibration Quality

**Problem**: Poor calibration results

**Solution**: 
1. Verify ideal responses are correct
2. Check measurement quality
3. Include switch terms
4. Try different calibration algorithms

## Resources

### References

- [networks.md](references/networks.md) - Network creation and manipulation
- [calibration.md](references/calibration.md) - Calibration methods and algorithms
- [media.md](references/media.md) - Transmission line media models
- [plotting.md](references/plotting.md) - Plotting and visualization
- [vna_integration.md](references/vna_integration.md) - VNA control and data acquisition
- [qfactor.md](references/qfactor.md) - Quality factor analysis

### Scripts

- [calibration_workflow.py](scripts/calibration_workflow.py) - Complete calibration workflow
- [network_analysis.py](scripts/network_analysis.py) - Network analysis tools
- [measurement_automation.py](scripts/measurement_automation.py) - VNA measurement automation

### External Resources

- Official documentation: https://scikit-rf.readthedocs.io/
- GitHub repository: https://github.com/scikit-rf/scikit-rf
- PyPI package: https://pypi.org/project/scikit-rf/
- Paper: "VNA Calibration", IEEE Microwave Magazine 2008