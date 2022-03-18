#! /bin/bash
if [ "$#" -ne "1" ] ;then
    echo "Please input a filename: s1ace2 or deltaace2!"
    exit 0
else
    FILENAME=$1
    echo "The input filename is $FILENAME. Begin the simulation!"
fi

mkdir -p min1
cd min1
cat > min1.in << EOF
S1 minimization
  mode = minimization
  step_limit = 5000
  write_information_interval = 1000
  dt = 1e-7
EOF
python ../../src/numpy/main_numpy.py --i ./min1.in --amber_parm ../../data/$FILENAME.parm7 --c ../../data/$FILENAME.rst7  --r $FILENAME\_min1.rst7
cd ..

mkdir -p min2
cd min2
cat > min2.in << EOF
S1 minimization
  mode = minimization
  step_limit = 5000
  write_information_interval = 1000
  dt = 1e-5
EOF
python ../../src/numpy/main_numpy.py --i ./min2.in --amber_parm ../../data/$FILENAME.parm7 --c ../min1/$FILENAME\_min1.rst7  --r $FILENAME\_min2.rst7
cd ..

mkdir -p min3
cd min3
cat > min3.in << EOF
S1 minimization
  mode = minimization
  step_limit = 5000
  write_information_interval = 1000
  dt = 1e-3
EOF
python ../../src/numpy/main_numpy.py --i ./min3.in --amber_parm ../../data/$FILENAME.parm7 --c ../min2/$FILENAME\_min2.rst7  --r $FILENAME\_min3.rst7
cd ..

mkdir -p heat
cd heat
cat > heat.in << EOF
S2 heat
  mode = nvt
  step_limit = 100000
  dt = 1e-3
  target_temperature = 300.0
  write_information_interval = 1000
  cutoff = 10.0
  thermostat = langevin_liu
EOF
python ../../src/numpy/main_numpy.py --i ./heat.in --amber_parm ../../data/$FILENAME.parm7 --c ../min3/$FILENAME\_min3.rst7  --r $FILENAME\_heat.rst7
cd ..

mkdir -p pres
cd pres
cat > pres.in << EOF
S3 press
  mode = npt
  step_limit = 200000
  dt = 1e-3
  constrain_mode = simple_constrain
  target_temperature = 300.0
  target_pressure = 1.0
  write_information_interval = 2500
  cutoff = 10.0
  thermostat = langevin_liu
  barostat = berendsen
EOF
python ../../src/numpy/run_npt_numpy.py --i ./pres.in --amber_parm ../../data/$FILENAME.parm7 --c ../heat/$FILENAME\_heat.rst7  --r $FILENAME\_press.rst7
cd ..

mkdir -p eq
cd eq
cat > eq.in << EOF
S4 eq
  mode = npt
  step_limit = 5000000
  dt = 2e-3
  constrain_mode = simple_constrain
  target_temperature = 300.0
  target_pressure = 1.0
  write_information_interval = 1000
  cutoff = 10.0
  thermostat = langevin_liu
  barostat = berendsen
EOF
python ../../src/numpy/run_npt_numpy.py --i ./eq.in --amber_parm ../../data/$FILENAME.parm7 --c ../pres/$FILENAME\_press.rst7  --r $FILENAME\_eq.rst7
cd ..

mkdir -p product
cd product
cat > md.in << EOF
S4 product
  mode = npt
  step_limit = 10000000
  dt = 4e-3
  constrain_mode = simple_constrain
  target_temperature = 300.0
  write_information_interval = 1000
  write_restart_file_interval = 250000
  target_pressure = 1.0
  thermostat = langevin_liu
  cutoff = 10.0
  barostat = berendsen
EOF
python ../../src/numpy/run_npt_numpy.py --i ./md.in --amber_parm ../../data/$FILENAME.parm7 --c ../eq/$FILENAME\_eq.rst7  --r $FILENAME\_md1.rst7
cd ..
