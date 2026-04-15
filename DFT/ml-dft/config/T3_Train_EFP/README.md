# Retraining the potential energy, forces and stress model and making predictions using new weights.

The ML-DFT package allows to retrain the model for the potential energy, forces and stresses, starting from the weights used in the paper. To do this you will need, for each structure, the POSCAR file as well as: (1) a file named *energy* containing the total potential energy (in eV); (2) a file named *forces* containing the atomic forces (in eV/A); and (3) a file named *stress* containing the six independent stress components in kB in the order XX,YY,ZZ,XY,YZ,ZX.


(1) In the files Train.csv and Val.cs write the locations of the training and validation structures where the POSCAR, energy, forces, and stress files are located.

(2) In the file *inp_params.py* you need to set *train_e=True*, and select the amount of epochs *ert_epochs*, the batch size *ert_batch_size*, and number of epochs without improvement before stopping *ert_patience*. Also, if you want to then make predictions of the energy, forces, and stress using the new weights set *test_e=True*. 

(3) One important note: if you set *train_e=True* and *test_e=True*, the program will automatically use the new weights from the retraining for the prediction. However, if you want to do first the training and then run again the program to perform predictions using the NEW weights, since for only making predictions you will have *train_e=False*, you need to set *new_weights_e=True* to let the program now it needs to use the new weights created. Otherwise, it will use the weights from the paper by default.

(4) For the predictions, you also need to have the *predict.csv* file.

Once all tags are selected, just do:

```angular2
python ML_DFT.py
```
