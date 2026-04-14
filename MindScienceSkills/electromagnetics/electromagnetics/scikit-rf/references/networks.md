# Networks in scikit-rf

Networks are the central objects in scikit-rf, representing N-port microwave networks.

## Creating Networks

### From Touchstone Files

Load networks from Touchstone (.sNp) files:

```python
import skrf as rf

# Load single network
network = rf.Network('data/device.s2p')

# Load multiple networks
networks = rf.io.read_all('data/', contains='s2p')
```

### From S-parameters

Create networks from scattering parameters:

```python
import numpy as np
import skrf as rf

# Create frequency array
freq = rf.Frequency(1, 10, 101, 'GHz')

# Create S-parameter matrix
s = np.random.uniform(size=(101, 2, 2)) + 1j*np.random.uniform(size=(101, 2, 2))

# Create network
network = rf.Network(frequency=freq, s=s, name='my_network')
```

### From Z-parameters

Create networks from impedance parameters:

```python
# Create Z-parameter matrix
z = np.full((101, 2, 2), 50+0j)  # 50 Ohm impedance

# Create network
network = rf.Network(frequency=freq, z=z)
```

### From Other Parameters

Create networks from Y, ABCD, T, or H parameters:

```python
# From ABCD parameters
abcd = np.array([[1, 50], [0, 1]])
a = np.tile(abcd, (101, 1, 1))
network = rf.Network(frequency=freq, a=a)

# From T-parameters
t = np.random.uniform(size=(101, 2, 2)) + 1j*np.random.uniform(size=(101, 2, 2))
network = rf.Network(frequency=freq, t=t)
```

## Network Properties

### Basic Properties

```python
# S-parameters
print(network.s)  # shape: (nfreq, nports, nports)

# Z-parameters
print(network.z0)  # port impedance

# Frequency
print(network.frequency)  # Frequency object
print(network.frequency.f)  # frequency array
print(network.frequency.f_scaled)  # scaled to 1.0 GHz
```

### S-parameter Components

```python
# Individual S-parameters
network.s11  # S11
network.s12  # S12
network.s21  # S21
network.s22  # S22

# Magnitude and phase
network.s_mag  # magnitude
network.s_deg  # phase in degrees
network.s_rad  # phase in radians

# Real and imaginary parts
network.s_re  # real part
network.s_im  # imaginary part
```

### Z-parameter Components

```python
# Z-parameter matrix
print(network.z)  # shape: (nfreq, nports, nports)

# Z-parameter components
network.z_re  # real part
network.z_im  # imaginary part
network.z_mag  # magnitude
network.z_ang  # angle
```

### Other Parameters

```python
# Y-parameters (admittance)
network.y

# ABCD-parameters
network.a

# T-parameters (transfer scattering)
network.t

# H-parameters (hybrid)
network.h
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

# Power
squared_network = network1 ** 2

# Negation
negated_network = -network1
```

### Scalar Operations

Operations with scalars and arrays:

```python
# Multiply by scalar
scaled_network = network1 * 2.0

# Add scalar
shifted_network = network1 + 1.0

# Multiply by frequency-dependent array
freq_array = np.linspace(1, 2, 101)
scaled_network = network1 * freq_array[:, np.newaxis, np.newaxis]
```

### Network Comparison

Compare networks:

```python
# Equality
if network1 == network2:
    print("Networks are equal")

# Inequality
if network1 != network2:
    print("Networks are different")
```

## Network Slicing

### Frequency Slicing

Slice networks by frequency:

```python
# Slice by frequency range
network_subset = network['1-5ghz']  # 1-5 GHz

# Slice by index
network_subset = network[0:50]  # first 50 points

# Slice by human-readable string
network_subset = network['80-90ghz']  # 80-90 GHz
```

### Parameter Slicing

Slice network parameters:

```python
# Slice S-parameters
s11_subset = network.s11[:10]  # first 10 frequency points
s11_port0 = network.s[:, 0, :]  # all parameters for port 0

# Slice Z-parameters
z_subset = network.z[50:100]  # frequency 50-100
```

## Network Cascading

### Two-Port Cascading

Cascade two-port networks:

```python
# Load two-port networks
line = rf.Network('data/line.s2p')
short = rf.Network('data/short.s2p')

# Cascade networks
cascaded = line ** short

# Plot result
cascaded.s21.plot_s_db()
plt.show()
```

### De-embedding

De-embed a network:

```python
# De-embed short from line
deembedded = line.inv ** short

# Verify
deembedded == short  # True
```

### Multi-Port Cascading

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

## Network Connections

### Port Connection

Connect specific ports between networks:

```python
# Load networks
tee = rf.Network('data/tee.s3p')
load = rf.Network('data/load.s2p')

# Connect port 1 of tee to port 0 of load
connected = rf.network.connect(tee, 1, load, 0)

# Plot result
connected.s21.plot_s_db()
plt.show()
```

### Arbitrary Connections

Connect arbitrary ports:

```python
# Connect multiple ports
connected = rf.network.connect(
    network1, (0, 1),  # ports 0 and 1 of network1
    network2, (0,),      # port 0 of network2
    network3, (2, 3)      # ports 2 and 3 of network3
)
```

## Network Interpolation

### Frequency Interpolation

Interpolate networks to different frequencies:

```python
# Load networks with different frequencies
network1 = rf.Network('data/network1.s2p')  # 101 points
network2 = rf.Network('data/network2.s2p')  # 201 points

# Resample network1 to 201 points
network1.resample(201)

# Now networks can be combined
combined = network1 + network2
```

### Network Interpolation

Interpolate network to specific frequencies:

```python
# Create new frequency array
new_freq = rf.Frequency(2, 8, 151, 'GHz')

# Interpolate network
interpolated = network.interpolate(new_freq)
```

## Network Concatenation

### Stitching Networks

Combine networks covering different frequency ranges:

```python
# Load networks
network_low = rf.Network('data/low_band.s2p')  # 1-5 GHz
network_high = rf.Network('data/high_band.s2p')  # 5-10 GHz

# Stitch networks
combined = rf.network.stitch(network_low, network_high)

# Plot combined response
combined.s21.plot_s_db()
plt.show()
```

## Network I/O

### Reading Networks

Read networks from files:

```python
# Read Touchstone file
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

## Network Utilities

### Parameter Conversion

Convert between parameter types:

```python
# Convert S to Z
z = rf.network.s2z(network.s)

# Convert Z to S
s = rf.network.z2s(network.z)

# Convert S to Y
y = rf.network.s2y(network.s)

# Convert Y to S
s = rf.network.y2s(network.y)

# Convert A to S
s = rf.network.a2s(network.a)

# Convert T to S
s = rf.network.t2s(network.t)
```

### Network Properties

Calculate network properties:

```python
# Check if network is reciprocal
is_reciprocal = network.is_reciprocal()

# Check if network is symmetric
is_symmetric = network.is_symmetric()

# Check if network is lossless
is_lossless = network.is_lossless()

# Get number of ports
nports = network.nports

# Get number of frequency points
nfreq = network.nfreq
```

## Best Practices

### Network Creation

1. **Use descriptive names**: `network = rf.Network('data/lowpass_filter.s2p')`
2. **Specify frequency explicitly**: `freq = rf.Frequency(1, 10, 101, 'GHz')`
3. **Include port impedance**: `z0=50` for 50 Ohm systems

### Network Operations

1. **Check frequency compatibility**: Ensure networks have matching frequencies
2. **Verify port counts**: Check that networks have compatible port counts
3. **Use appropriate operations**: Use `**` for cascading, `+` for combining

### Data Management

1. **Use Touchstone for long-term storage**: `.s2p` files are standard
2. **Use pickle for temporary storage**: `.ntwk` files for intermediate results
3. **Organize files by function**: Separate networks by type and purpose

## Troubleshooting

### Frequency Mismatch

**Problem**: `IndexError: Networks must have same frequency`

**Solution**: Resample or interpolate networks
```python
network1.resample(len(network2.frequency.f))
```

### Port Count Mismatch

**Problem**: Port count incompatibility

**Solution**: Check network port counts
```python
print(f"Network1 ports: {network1.nports}")
printouts: {network2.nports}")
```

### Invalid Network Data

**Problem**: NaN or Inf values in network parameters

**Solution**: Check for invalid values
```python
import numpy as np
if np.any(np.isnan(network.s)):
    print("Network contains NaN values")
if np.any(np.isinf(network.s)):
    print("Network contains Inf values")
```