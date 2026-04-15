# Retraining the DOS and VBM/CBM model and making predictions using new weights.

The ML-DFT package allows to retrain the model for the DOS, VBM, and CBM starting from the weights used in the paper. To do this, you will need, for each structure, the POSCAR file as well as: (1) a file named *dos* containing the dos (shifted with respect to the vacuum energy) written every 0.1 eV starting from -35 eV; and (2) a file named *VB_CB* containing the values of the VBM and CBM in eV (again, shifted with respect to the vacuum energy). For more info on the shift with respect to the vacuum energy, please refer to the ML-DFT publication.

(1) In the files Train.csv and Val.cs write the locations of the training and validation structures where the POSCAR, dos and VB_CB files are located.

(2) In the file *inp_params.py* you need to set *train_dos=True*, and select the amount of epochs *drt_epochs*, the batch size *drt_batch_size*, and number of epochs without improvement before stopping *drt_patience*. Also, if you want to then make predictions of the dos and VBM and CBM using the new weights set *test_dos=True*. 

(3) As with the energy in tutorial T2, if you set *train_dos=True* and *test_dos=True*, the program will automatically use the new weights from the retraining for the prediction. However, if you want to do first the training and then run again the program to perform predictions using the NEW weights, since for only making predictions you will have *train_dos=False*, you need to set *new_weights_dos=True* to let the program now it needs to use the new weights created, not the weights from the paper.

(4) For the predictions, you also need to have the *predict.csv* file.

Once all tags are selected, just do:

```angular2
python ML_DFT.py
```

NOTE: While we do not include the crystal structures in the Train.csv and Val.csv files, as we do not have the energy value of vacuum to shift the dos and VBM and CBM data, we do include them in the predict.csv file as predictions can be made on any type of structure.

