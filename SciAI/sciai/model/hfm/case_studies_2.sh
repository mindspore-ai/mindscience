#!/bin/bash

FAIL=0

mkdir ./data/results

python train.py --t 26 --n 2500 > ./data/results/Cylinder2D_flower_26_2500_stdout.txt &
python train.py --t 13 --n 2500 > ./data/results/Cylinder2D_flower_13_2500_stdout.txt &
python train.py --t 7 --n 2500 > ./data/results/Cylinder2D_flower_7_2500_stdout.txt &
python train.py --t 3 --n 2500 > ./data/results/Cylinder2D_flower_3_2500_stdout.txt &

# python train.py --t 26 --n 1500 > ./data/results/Cylinder2D_flower_26_1500_stdout.txt &
# python train.py --t 13 --n 1500 > ./data/results/Cylinder2D_flower_13_1500_stdout.txt &
# python train.py --t 7 --n 1500 > ./data/results/Cylinder2D_flower_7_1500_stdout.txt &
# python train.py --t 3 --n 1500 > ./data/results/Cylinder2D_flower_3_1500_stdout.txt &
 
# python train.py --t 26 --n 500 > ./data/results/Cylinder2D_flower_26_500_stdout.txt &
# python train.py --t 13 --n 500 > ./data/results/Cylinder2D_flower_13_500_stdout.txt &
# python train.py --t 7 --n 500 > ./data/results/Cylinder2D_flower_7_500_stdout.txt &
# python train.py --t 3 --n 500 > ./data/results/Cylinder2D_flower_3_500_stdout.txt &
 
# python train.py --t 26 --n 250 > ./data/results/Cylinder2D_flower_26_250_stdout.txt &
# python train.py --t 13 --n 250 > ./data/results/Cylinder2D_flower_13_250_stdout.txt &
# python train.py --t 7 --n 250 > ./data/results/Cylinder2D_flower_7_250_stdout.txt &
# python train.py --t 3 --n 250 > ./data/results/Cylinder2D_flower_3_250_stdout.txt &

for job in `jobs -p`
do
    echo $job
    wait $job || let "FAIL+=1"
done

echo $FAIL

if [ "$FAIL" == "0" ];
then
    echo "YAY"
else
    echo "FAIL! ($FAIL)"
fi
