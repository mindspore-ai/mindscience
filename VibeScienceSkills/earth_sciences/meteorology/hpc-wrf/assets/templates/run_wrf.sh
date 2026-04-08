#!/bin/bash
# Complete WRF run script

# Configuration
WRF_DIR=/path/to/WRF
WPS_DIR=/path/to/WPS
RUN_DIR=/path/to/run_directory
GEOG_DIR=/path/to/WPS_GEOG
GRIB_DIR=/path/to/grib_data

# Number of MPI tasks
NTASKS=128

# Simulation dates
START_DATE="2024-01-01_00:00:00"
END_DATE="2024-01-03_00:00:00"

# Load modules
module load intel impi netcdf

# Step 1: Run geogrid
echo "=== Running geogrid ==="
cd $RUN_DIR
ln -sf $WPS_DIR/geogrid .
ln -sf $WPS_DIR/metgrid .
ln -sf $WPS_DIR/ungrib .
ln -sf $WPS_DIR/geogrid/METALES.TBL .
ln -sf $WPS_DIR/*.TBL .

# Copy namelist
cp namelist.wps $RUN_DIR/

./geogrid.exe
if [ $? -ne 0 ]; then
    echo "geogrid.exe failed"
    exit 1
fi

# Step 2: Run ungrib
echo "=== Running ungrib ==="
ln -sf $WPS_DIR/ungrib/Variable_Tables/Vtable.GFS Vtable
./link_grib.csh $GRIB_DIR/gfs.*

./ungrib.exe
if [ $? -ne 0 ]; then
    echo "ungrib.exe failed"
    exit 1
fi

# Step 3: Run metgrid
echo "=== Running metgrid ==="
./metgrid.exe
if [ $? -ne 0 ]; then
    echo "metgrid.exe failed"
    exit 1
fi

# Step 4: Run real
echo "=== Running real ==="
cd $RUN_DIR
ln -sf $WRF_DIR/run/* .
cp namelist.input $RUN_DIR/

./real.exe
if [ $? -ne 0 ]; then
    echo "real.exe failed"
    exit 1
fi

# Check for wrfinput and wrfbdy files
if [ ! -f "wrfinput_d01" ] || [ ! -f "wrfbdy_d01" ]; then
    echo "Error: wrfinput_d01 or wrfbdy_d01 not created"
    exit 1
fi

# Step 5: Run wrf
echo "=== Running wrf ==="
mpirun -np $NTASKS ./wrf.exe

# Check for success
if grep -q "SUCCESS COMPLETE WRF" rsl.out.0000; then
    echo "=== WRF completed successfully ==="
else
    echo "=== WRF may have failed ==="
    exit 1
fi
