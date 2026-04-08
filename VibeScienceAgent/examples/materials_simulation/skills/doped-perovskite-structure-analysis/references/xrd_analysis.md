# XRD Pattern Analysis Reference

Advanced techniques for analyzing and comparing XRD patterns from DFT-optimized structures.

## XRDCalculator Configuration

### Radiation Sources

```python
from pymatgen.analysis.diffraction.xrd import XRDCalculator

# Cu Kα (default)
xrd_calc = XRDCalculator(wavelength="CuKa")  # 1.54184 Å

# Other common sources
xrd_calc = XRDCalculator(wavelength="MoKa")  # 0.71073 Å
xrd_calc = XRDCalculator(wavelength="CoKa")  # 1.78897 Å
xrd_calc = XRDCalculator(wavelength="FeKa")  # 1.93728 Å

# Custom wavelength
xrd_calc = XRDCalculator(wavelength=1.5406)  # in Angstroms
```

### Calculation Parameters

```python
pattern = xrd_calc.get_pattern(
    structure,
    two_theta_range=(10, 90),  # 2θ range in degrees
    cap_thick=0.5               # Capillary thickness for Debye-Scherrer
)
```

## Pattern Analysis

### Extract Peak Information

```python
# Access pattern data
two_theta = pattern.x      # 2θ angles
intensity = pattern.y      # Intensities
hkls = pattern.hkls        # Miller indices
d_hkl = pattern.d_hkl      # d-spacings

# Print main peaks
for i, (t, inten, hkl_list) in enumerate(zip(pattern.x, pattern.y, pattern.hkls)):
    if inten > 5:  # Filter by intensity
        print(f"2θ = {t:.2f}°, I = {inten:.1f}, hkl = {hkl_list}")
```

### Compare Multiple Patterns

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_xrd_comparison(structures, labels, wavelength="CuKa"):
    """Compare XRD patterns of multiple structures."""
    xrd_calc = XRDCalculator(wavelength=wavelength)
    
    plt.figure(figsize=(12, 6))
    colors = ['b', 'r', 'g', 'm', 'c']
    
    for i, (struct, label) in enumerate(zip(structures, labels)):
        pattern = xrd_calc.get_pattern(struct, two_theta_range=(10, 90))
        # Offset for visibility
        offset = i * 100
        plt.plot(pattern.x, pattern.y + offset, 
                color=colors[i % len(colors)], 
                label=label, linewidth=1.5)
    
    plt.xlabel('2θ (degrees)', fontsize=14)
    plt.ylabel('Intensity (a.u.)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    return plt

# Usage
structures = [structure_pure, structure_doped]
labels = ['Pure SrTiO3', 'Ba-doped SrTiO3']
plot_xrd_comparison(structures, labels)
plt.savefig('xrd_comparison.png', dpi=300)
```

## Peak Shift Analysis

### Calculate Lattice Parameter Change

```python
def analyze_peak_shift(pattern_before, pattern_after, hkl_index=0):
    """Analyze peak shift between two patterns."""
    # Get the main peak position
    theta_before = pattern_before.x[hkl_index]
    theta_after = pattern_after.x[hkl_index]
    
    shift = theta_after - theta_before
    
    # Calculate d-spacing change
    import math
    wavelength = 1.54184  # Cu Kα
    
    d_before = wavelength / (2 * math.sin(math.radians(theta_before / 2)))
    d_after = wavelength / (2 * math.sin(math.radians(theta_after / 2)))
    
    d_change = (d_after - d_before) / d_before * 100
    
    print(f"Peak shift: {shift:.3f}°")
    print(f"d-spacing change: {d_change:.2f}%")
    
    return shift, d_change
```

### Ionic Radius Effect

| Dopant | Host Site | Ionic Radius (Å) | Expected Effect |
|--------|-----------|------------------|-----------------|
| Ba²⁺ | Sr²⁺ | 1.61 > 1.44 | Lattice expansion, peaks shift to lower angles |
| Ca²⁺ | Sr²⁺ | 1.18 < 1.44 | Lattice contraction, peaks shift to higher angles |
| La³⁺ | Sr²⁺ | 1.36 < 1.44 | Slight contraction |

## Symmetry Analysis

### Detect Phase Transitions

```python
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

def check_symmetry_change(structure_initial, structure_relaxed):
    """Check for symmetry changes after relaxation."""
    analyzer_initial = SpacegroupAnalyzer(structure_initial)
    analyzer_relaxed = SpacegroupAnalyzer(structure_relaxed)
    
    sg_initial = analyzer_initial.get_space_group_number()
    sg_relaxed = analyzer_relaxed.get_space_group_number()
    
    print(f"Initial space group: {sg_initial}")
    print(f"Relaxed space group: {sg_relaxed}")
    
    if sg_initial != sg_relaxed:
        print("⚠️ Symmetry change detected!")
        print(f"  {analyzer_initial.get_space_group_symbol()} → {analyzer_relaxed.get_space_group_symbol()}")
    
    return sg_initial, sg_relaxed
```

### Common Phase Transitions in SrTiO3

| Doping | Possible Transition | Peak Signature |
|--------|---------------------|----------------|
| Heavy doping | Cubic → Tetragonal | Peak splitting at ~46° (200) |
| Strain | Cubic → Orthorhombic | Multiple peak splitting |

## Export Data

### Save Pattern Data

```python
import pandas as pd

def save_xrd_data(pattern, filename):
    """Save XRD pattern to CSV."""
    data = {
        'two_theta': pattern.x,
        'intensity': pattern.y,
        'd_spacing': pattern.d_hkl,
        'hkl': [str(h) for h in pattern.hkls]
    }
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"XRD data saved to {filename}")

# Usage
save_xrd_data(pattern, 'xrd_data.csv')
```

### Export for Rietveld Refinement

```python
def export_for_rietveld(pattern, filename, wavelength=1.54184):
    """Export in format suitable for Rietveld refinement software."""
    import numpy as np
    
    # Create high-resolution pattern
    two_theta = np.linspace(10, 90, 8001)
    intensity = np.zeros_like(two_theta)
    
    # Add peaks with Gaussian broadening
    for t, inten in zip(pattern.x, pattern.y):
        # FWHM ~ 0.1° typical for lab XRD
        sigma = 0.1 / 2.355
        peak = inten * np.exp(-(two_theta - t)**2 / (2 * sigma**2))
        intensity += peak
    
    # Save
    with open(filename, 'w') as f:
        f.write(f"# XRD Pattern\n")
        f.write(f"# Wavelength: {wavelength} Å\n")
        for t, i in zip(two_theta, intensity):
            f.write(f"{t:.4f}  {i:.4f}\n")
```
