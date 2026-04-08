# VASP Material-Specific Parameters

## Contents

- Magnetic materials
- Semiconductors and insulators
- Metals
- Strongly correlated systems (LDA+U)
- DFT+U values for common elements
- K-point density guidelines

## Magnetic Materials

Enable spin polarization and set initial magnetic moments:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `ISPIN` | 2 | Spin polarization enabled |
| `MAGMOM` | per-atom values | One value per atom in POSCAR order |
| `ISIF` | 3 | Required for stress calculation with magnetism |

MAGMOM format: `MAGMOM = m1 m2 m3 ...` (one value per atom)

Common initial magnetic moments:
- Fe: 5.0 µB
- Co: 3.0 µB
- Ni: 2.0 µB
- Mn: 3.0–5.0 µB
- Cr: 3.0–5.0 µB

If the material is magnetic or may break spin symmetry, the initial magnetic setup determines whether SCF converges to a physically meaningful state. Use a trustworthy initial guess rather than zero.

## Semiconductors and Insulators

Use Gaussian smearing for accurate band gaps:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `ISMEAR` | 0 | Gaussian smearing |
| `SIGMA` | 0.05 | Smearing width |
| `EDIFF` | 1E-6 | Tight convergence for accurate gap |

The default `ISMEAR=1` is not appropriate for insulators and semiconductors. Tetrahedron methods (ISMEAR=-5) also work well for static DOS when using a Gamma-centered mesh.

For **accurate band gap with hybrid functionals**:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `LHFCALC` | .TRUE. | Enable hybrid functional |
| `AEXX` | 0.25 | Hartree-Fock exchange ratio (PBE0) |
| `ENCUT` | ≥500 | Ensure basis convergence |

## Metals

Use metallic smearing for partial occupancies:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `ISMEAR` | 1 or 2 | Methfessel-Paxton 1st or 2nd order |
| `SIGMA` | 0.1–0.2 | Broader smearing for metals |
| `EDIFF` | 1E-5 | Standard electronic convergence |

## Strongly Correlated Systems (LDA+U)

For transition metal oxides and f-electron systems (NiO, CeO2, Fe3O4, etc.):

| Parameter | Value | Notes |
|-----------|-------|-------|
| `LDAU` | .TRUE. | Enable LDA+U |
| `LDAUTYPE` | 2 | Dudarev approach (only U-J matters) |
| `LDAUU` | per-element | U value for each element (eV) |
| `LDAUJ` | per-element | J value for each element (eV) |
| `LDAUL` | 2 for d, 3 for f | Angular momentum channel |

In Dudarev's formulation, only the difference `LDAUU - LDAUJ` is meaningful.

## DFT+U Values for Common Elements

Typical U values (in eV) for strongly correlated materials:

| Element | Channel | Typical U (eV) | Reference |
|---------|---------|---------------|-----------|
| Ni | 3d | 4.0–6.0 | PRB 57, 1505 (1998) |
| Co | 3d | 3.0–4.0 | PRB 63, 174110 (2001) |
| Fe | 3d | 3.0–4.0 | PRB 72, 144415 (2005) |
| Mn | 3d | 2.5–3.5 | PRB 70, 094425 (2004) |
| Cr | 3d | 2.5–3.5 | PRB 57, 1505 (1998) |
| V | 3d | 2.0–3.0 | PRB 62, 11556 (2000) |
| Ce | 4f | 4.0–6.0 | PRB 75, 035113 (2007) |

## K-Point Density Guidelines

### KSPACING approach

| System | KSPACING | Comment |
|--------|----------|---------|
| Metals | 0.20–0.26 | Dense for good BZ sampling |
| Semiconductors | 0.20–0.30 | Balance accuracy and cost |
| Insulators | 0.30–0.40 | Less dense needed |

### Traditional KPOINTS mesh

| Goal | Typical KPOINTS mode |
|------|---------------------|
| routine SCF on periodic bulk | Monkhorst-Pack or Gamma-centered mesh |
| large supercell or Gamma-only workflow | Gamma-only file |
| band path | line-mode path |

For DOS: use a dense uniform mesh (not a symmetry path). For band structure: use line-mode k-points along high-symmetry paths.

## Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `EDIFFG not reached` | Insufficient ionic steps | Increase `NSW` |
| `TOO FEW BANDS` | Not enough empty bands | Increase `NBANDS` by 20–50% |
| `EDDDAV: ZHEGV failed` | Memory/algorithm issue | Use `ALGO = Normal`, increase `NCORE` |
| `BRMIX: very serious` | Charge density mixing divergence | Increase `IMIX`, decrease `AMIX`, or use `ALGO = All` |

## References

1. VASP Official Tutorial: https://www.vasp.at/wiki/
2. Materials Project DFT Guidelines: https://materialsproject.org
3. qvasp Toolkit: https://qvasp.com
4. VASPKIT: https://vaspkit.com (CPC 267, 108033)
