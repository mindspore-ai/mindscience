# Data Generation

Python dependencies:

```bash
pip3 install h5py matplotlib
```

Data generation utilizes the Dedalus package. Follow the installation instructions [here](https://dedalus-project.readthedocs.io/en/latest/pages/installation.html).
Once installation is complete, you can run script files to generate data. For example, use the following command to generate a pre-training dataset with coefficient range $[-3,3]$ and random seed 1:

```bash
OMP_NUM_THREADS=1 python3 custom_sinus.py -m 3 -r 1
```

You can use the following command to view the list of command-line arguments:

```bash
python3 custom_sinus.py -h
```

## File Directory

```text
./
│  common.py                                 # Common utilities for data generation
│  custom_multi_component.py                 # Main file for generating multi-component equation data
│  custom_sinus.py                           # Main file for generating equation data with sine/cosine function terms
│  custom_wave.py                             # Main file for generating wave equation data
│  inverse_sinus.py                          # Main file for generating inverse problem data for equations with sine/cosine function terms
│  inverse_wave.py                           # Main file for generating wave equation inverse problem data
│  README.md                                 # Dataset generation documentation (English)
│  README_CN.md                              # Dataset generation documentation (Chinese)
```
