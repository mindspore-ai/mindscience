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
  step_limit = 1000
  write_information_interval = 100
  dt = 1e-7
EOF
python ../../../examples/covid/src/numpy/main_numpy.py --i ./min1.in --amber_parm ../data/$FILENAME.parm7 --c ../data/$FILENAME.rst7  --r $FILENAME\_min1.rst7
cd ..


mkdir -p pres
cd pres
cat > pres.in << EOF
S3 press
  mode = npt
  step_limit = 1000
  dt = 1e-3
  constrain_mode = simple_constrain
  target_temperature = 300.0
  target_pressure = 1.0
  write_information_interval = 100
  cutoff = 10.0
  thermostat = langevin_liu
  barostat = berendsen
EOF
python ../../../examples/covid/src/numpy/run_npt_numpy.py --i ./pres.in --amber_parm ../data/$FILENAME.parm7 --c ../data/$FILENAME\_heat.rst7  --r $FILENAME\_press.rst7
cd ..

