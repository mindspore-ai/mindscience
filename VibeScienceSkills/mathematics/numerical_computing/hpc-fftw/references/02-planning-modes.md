# FFTW Planning Modes

## Planning Flags

| Flag | Planning Time | Execution Speed | Use Case |
|------|---------------|-----------------|----------|
| FFTW_ESTIMATE | ~0 | Baseline | One-time transforms |
| FFTW_MEASURE | Moderate | Good | Default, repeated transforms |
| FFTW_PATIENT | Slow | Better | Production runs |
| FFTW_EXHAUSTIVE | Very slow | Best | Critical performance |

## Planning Process

1. **ESTIMATE**: Uses heuristics, no timing
   - Fastest planning
   - Suboptimal execution
   - Safe for one-time use

2. **MEASURE**: Times multiple algorithms
   - Moderate planning time
   - Good execution speed
   - Default choice

3. **PATIENT**: Explores more algorithms
   - Slow planning (seconds to minutes)
   - Better execution
   - Use with wisdom files

4. **EXHAUSTIVE**: Tests all algorithms
   - Very slow planning (minutes to hours)
   - Optimal execution
   - Production-critical applications

## Wisdom System

### Save Wisdom
```c
fftw_export_wisdom_to_filename("wisdom.dat");
// Or to string
char *wisdom = fftw_export_wisdom_to_string();
```

### Load Wisdom
```c
fftw_import_wisdom_from_filename("wisdom.dat");
// Or from string
fftw_import_wisdom_from_string(wisdom_string);
```

### System-wide Wisdom
```bash
# Set wisdom file location
export FFTW_WISDOM_ONLY=1  # Only use wisdom, no planning
```

## Best Practices

1. **Development**: Use FFTW_ESTIMATE for quick testing
2. **Benchmarking**: Use FFTW_MEASURE
3. **Production**: Use FFTW_PATIENT with wisdom files
4. **Critical paths**: Use FFTW_EXHAUSTIVE, save wisdom
