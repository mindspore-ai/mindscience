# OpenBLAS Compilation

## Building from Source

### Basic Build
```bash
git clone https://github.com/OpenMathLib/OpenBLAS.git
cd OpenBLAS
make -j8
sudo make install PREFIX=/usr/local
```

### Build Options
```bash
# Enable OpenMP threading
make USE_OPENMP=1

# Target specific CPU
make TARGET=HASWELL

# Build 64-bit interface (ILP64)
make INTERFACE64=1

# Include LAPACK
make NO_LAPACK=0

# Build shared library
make NO_STATIC=1

# Build static library
make NO_SHARED=1
```

### Cross-Compilation
```bash
# For ARM
make CC=arm-linux-gnueabihf-gcc FC=arm-linux-gnueabihf-gfortran TARGET=ARMV8

# For specific architecture
make TARGET=SKYLAKEX BINARY=64
```

## Linking

### Static Linking
```bash
gcc -o myprogram myprogram.c -lopenblas -lpthread -lm
```

### Dynamic Linking
```bash
gcc -o myprogram myprogram.c -L/usr/local/lib -lopenblas
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

### With LAPACK
```bash
# OpenBLAS includes LAPACK
gcc -o myprogram myprogram.c -lopenblas

# Or with separate LAPACK
gcc -o myprogram myprogram.c -lopenblas -llapack
```

## CMake Integration

```cmake
find_package(OpenBLAS REQUIRED)

add_executable(myprogram myprogram.c)
target_link_libraries(myprogram OpenBLAS::OpenBLAS)
```

## pkg-config

```bash
# Compile
gcc -o myprogram myprogram.c $(pkg-config --cflags openblas)

# Link
gcc -o myprogram myprogram.c $(pkg-config --libs openblas)
```

## Common Build Targets

| Target | Architecture |
|--------|--------------|
| HASWELL | Intel Haswell+ |
| SKYLAKEX | Intel Skylake-X (AVX-512) |
| ZEN | AMD Zen |
| ARMV8 | ARMv8 (AArch64) |
| POWER8 | IBM POWER8 |
| POWER9 | IBM POWER9 |
