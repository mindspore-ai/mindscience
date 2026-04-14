# SCF Convergence

## What is SCF?

Self-Consistent Field is the iterative solution of the Hartree-Fock or DFT
equations. Electrons find their optimal distribution in the field of nuclei
and other electrons.

## Why SCF Fails

SCF may fail to converge when:
- Starting guess is poor
- Geometry is far from equilibrium
- System has pathological electronic structure (degenerate states, diradicals)
- Basis set is inappropriate

## SCF Options

### Density Guess

```gjf
#p B3LYP/6-311G(d,p) Guess=TCore
```

| Guess | Meaning | Use |
|-------|---------|-----|
| `Guess=Core` | Core Hamiltonian guess | Default, fast |
| `Guess=TCore` | Tight core guess | Better for difficult cases |
| `Guess=Mix` | Mixed atomic orbitals | For open-shell or diradicals |
| `Guess=Fragment` | Fragment-based guess | Large systems |

### SCF Cycles

```gjf
#p B3LYP/6-311G(d,p) SCF=Conver=8
```

Increase max cycles if needed:

```gjf
#p B3LYP/6-311G(d,p) SCF=maxcycles=500
```

## Convergence Problems by System Type

### Closed Shell (singlet, all paired)

1. Try `Guess=TCore`
2. Try `SCF=Conver=8` (tighter convergence)
3. Try different functional (B3LYP → PBE0)

### Open Shell (doublet, radical)

```gjf
#p UB3LYP/6-311G(d,p) Guess=Mix
```

Use unrestricted (U) for open-shell systems.

### Diradicals and Near-Degenerate States

For systems with two similar energy states:

```gjf
#p UM06-2X/6-311G(d,p) Guess=Mix
```

Or use multi-reference methods (CASSCF, NEVPT2).

## DIIS (Direct Inversion in Iterative Subspace)

DIIS is used by default to accelerate convergence. For difficult cases:

```gjf
#p B3LYP/6-311G(d,p) SCF=DIIS
```

To disable DIIS (rarely needed):

```gjf
#p B3LYP/6-311G(d,p) SCF=NoDIIS
```

## Level-Shifting

For oscillatory convergence:

```gjf
#p B3LYP/6-311G(d,p) SCF=LevelShift=0.5
```

Level shifting adds an energy penalty to occupied orbitals to prevent
oscillations.

## Stability Checks

If SCF converges but to the wrong solution:

```gjf
#p B3LYP/6-311G(d,p) Stable=Opt
```

This checks wavefunction stability and corrects if needed.

## Common SCF Error Messages

| Message | Meaning | Fix |
|---------|---------|-----|
| `SCF not converged` | Didn't reach self-consistency | Increase cycles, use better guess |
| `Lowadfadf` | Degenerate states | Use `Guess=Mix` or multi-reference |
| `Bad convergence` | Geometry too far from optimum | Optimize geometry first |
| `Mayer indices bad` | Basis set issues | Check basis set |

## Restarting from Checkpoint

If SCF converges slowly on a large system:

```gjf
%OldChk=previous.chk
#p B3LYP/6-311G(d,p) Guess=Read
```

`Guess=Read` uses the density from the checkpoint file as the starting guess.

## Workflow for Difficult SCF

```
1. Try: B3LYP/6-31G(d,p) Guess=TCore
2. If fails: Try Guess=Core (default) with tight SCF
3. If oscillates: Try SCF=LevelShift=0.3
4. If open-shell: Try Guess=Mix
5. If still fails: Try different functional or method
```

## Functional Recommendations for Difficult Cases

| System | Try This Functional |
|--------|---------------------|
| Standard organic | B3LYP |
| Non-covalent | ωB97X-D |
| Charge transfer | ωB97X-D |
| Transition metals | PBE0 or TPSSh |
| Difficult radicals | M06-2X |

## Testing SCF Convergence

Before running expensive calculations:

1. Test on a smaller basis set (6-31G)
2. Test with a faster functional (PBE)
3. Confirm convergence on a simpler system
