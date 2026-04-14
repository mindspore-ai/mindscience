# Predicting the charge density, DOS, VBM, CBM, potential energy, forces and stress 

To only make predictions using the already trained models, follow these steps:

(1) In the file *predict.csv*, write the location on the POSCAR files with the structures you want to make the predictions on.

(2) In the file *inp_params.py* we need to set *test_chg=True*, *test_e=True*, and *test_dos=True*. When making predictions, the charge density is always predicted, as it is the first step of the ML-DFT model. If you only want to predict the atomic properties (energy, forces, and stress), you can set *test_dos=False*. Similarly, if you only want to predict the DOS and VBM/CBM, just set *test_e=False*

(3) There are various options regarding the writing out and plotting of the predicted data. First, if you have the DFT valence electron density of the structure and want to compare the predicted valence electron density with it, set *comp_chg=True*. Also, if you want to print out the predicted valence electron density using the same grid as the DFT reference, then do *write_chg=True* and *ref_chg=True*. However, if you do not have the DFT valence electron density and just want to write the predicted valence electron density using a specific grid spacing (e.g. 0.5) just set the following tags: *write_chg=True*, *ref_den=False*, and *grid_spacing=0.5*. It is important to note that the smaller the grid spacing, the longer it will take to write it. The grid spacing does not affect in any way the accuracy of the predicted charge density.
Second, if you want to plot the predicted DOS just set *plot_dos=True*. 

Once all tags are selected, just do:

```angular2
python ML_DFT.py
```
The predicted values of atomic charges, energy, forces, stress, VBM, CBM, and BG will be written in the output file OUT_DATA.

The predicted charge density, once written out, can be visualized by just copying the file to CHGCAR and opening it with VESTA.
