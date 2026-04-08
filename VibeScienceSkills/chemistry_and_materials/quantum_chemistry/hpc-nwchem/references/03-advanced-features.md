# NWChem Advanced Features

## Open-Shell Calculations

```
scf
 uhf
 doublet
end

task scf
```

## Solvent Model

```
cosmo
 solvent water
end

task dft
```

## Periodic Systems

```
set nwpw:mp_ngrid_cutoff 50.0
set nwpw:mp_ngrid_wavefunction 20.0

nwpw
 simulation_cell
   lattice_vectors
     10.0 0.0 0.0
     0.0 10.0 0.0
     0.0 0.0 10.0
 end
end

task band
```

## QM/MM

```
qmmm
 region qm
   basis
    * library 6-31g*
   end
 end
end

task qmmm
```

## Parallel Computing

```bash
mpirun -np 8 nwchem input.nw
```

## Output Control

```
print low
print medium
print high
```