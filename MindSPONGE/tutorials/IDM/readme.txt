pre-requisite
1. mindspore
2. matplotlib
3. mdshare: conda install -c conda-forge mdshare

1. enter codes/ directory
2. open data_process.ipynb notebook, and run through all cells to download the dataset.
3. execute training script by typing:
   python train.py --data_path <your_data_path> --resolution <resolution>
   e.g.:
   python train.py --data_path ../data/ --resolution 400
