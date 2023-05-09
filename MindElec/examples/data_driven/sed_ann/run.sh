#!/bin/bash

startTime=$(date +%Y%m%d-%H:%M:%S)
startTime_s=$(date +%s)

echo "ic Re"
python eval.py --param_dict_path "./model/model-ic_Re.ckpt" --predict_input_path "./data/Test_ic_50_0.6.csv" --predict_output_path "output_Mag_ic_50_0.6.csv"

echo "ic Im"
python eval.py --param_dict_path "./model/model-ic_Im.ckpt" --predict_input_path "./data/Test_ic_50_0.6.csv" --predict_output_path "output_Pha_ic_50_0.6.csv"

echo "ec Re"
python eval.py --param_dict_path "./model/model-ec_Re.ckpt" --predict_input_path "./data/Test_ec_50_0.6.csv" --predict_output_path "output_Mag_ec_50_0.6.csv"

echo "ec Im"
python eval.py --param_dict_path "./model/model-ec_Im.ckpt" --predict_input_path "./data/Test_ec_50_0.6.csv" --predict_output_path "output_Pha_ec_50_0.6.csv"

echo "cc Re"
python eval.py --param_dict_path "./model/model-cc_Re.ckpt" --predict_input_path "./data/Test_cc_50_0.6.csv" --predict_output_path "output_Mag_cc_50_0.6.csv"

echo "cc Im"
python eval.py --param_dict_path "./model/model-cc_Im.ckpt" --predict_input_path "./data/Test_cc_50_0.6.csv" --predict_output_path "output_Pha_cc_50_0.6.csv"


endTime=$(date +%Y%m%d-%H:%M:%S)
endTime_s=$(date +%s)
sumTime=$((endTime_s - startTime_s))
echo "$startTime ---> $endTime" "Total:$sumTime seconds"