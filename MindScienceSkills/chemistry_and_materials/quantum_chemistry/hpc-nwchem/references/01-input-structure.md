# NWChem Input File Structure

## Basic Structure

```
echo "task description"
start project_name

memory total 4 gb

geometry units angstrom
 atomic coordinates
end

basis
 basis set definition
end

method_module
 parameter_settings
end

task method task_type
```

## Geometry Definition

```
geometry units angstrom
 C    0.0000    0.0000    0.0000
 H    0.0000    0.0000    1.0890
end
```

## Basis Set Definition

```
basis
 * library 6-31g*
end

# or specify element
basis
 C library 6-31g*
 H library 6-31g
end
```

## Task Types

| Task | Description |
|------|-------------|
| energy | Single point energy |
| optimize | Geometry optimization |
| frequencies | Frequency calculation |
| dynamics | Molecular dynamics |
| property | Property calculation |