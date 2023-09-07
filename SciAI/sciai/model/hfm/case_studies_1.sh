#!/bin/bash

FAIL=0

mkdir ./data/results

python train.py --t 201 --n 15000 > ./data/results/Cylinder2D_flower_201_15000_stdout.txt &
python train.py --t 101 --n 15000 > ./data/results/Cylinder2D_flower_101_15000_stdout.txt &
python train.py --t 51 --n 15000 > ./data/results/Cylinder2D_flower_51_15000_stdout.txt &
python train.py --t 26 --n 15000 > ./data/results/Cylinder2D_flower_26_15000_stdout.txt &

# python train.py --t 201 --n 10000 > ./data/results/Cylinder2D_flower_201_10000_stdout.txt &
# python train.py --t 101 --n 10000 > ./data/results/Cylinder2D_flower_101_10000_stdout.txt &
# python train.py --t 51 --n 10000 > ./data/results/Cylinder2D_flower_51_10000_stdout.txt &
# python train.py --t 26 --n 10000 > ./data/results/Cylinder2D_flower_26_10000_stdout.txt &

# python train.py --t 201 --n 5000 > ./data/results/Cylinder2D_flower_201_5000_stdout.txt &
# python train.py --t 101 --n 5000 > ./data/results/Cylinder2D_flower_101_5000_stdout.txt &
# python train.py --t 51 --n 5000 > ./data/results/Cylinder2D_flower_51_5000_stdout.txt &
# python train.py --t 26 --n 5000 > ./data/results/Cylinder2D_flower_26_5000_stdout.txt &

# python train.py --t 201 --n 2500 > ./data/results/Cylinder2D_flower_201_2500_stdout.txt &
# python train.py --t 101 --n 2500 > ./data/results/Cylinder2D_flower_101_2500_stdout.txt &
# python train.py --t 51 --n 2500 > ./data/results/Cylinder2D_flower_51_2500_stdout.txt &
# python train.py --t 26 --n 2500 > ./data/results/Cylinder2D_flower_26_2500_stdout.txt &

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
