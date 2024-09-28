English | [简体中文](./README_CN.md)

## Table of Contents

- [Table of Contents](#table-of-contents)
- [GAD-MALL Description](#gad-mall-description)
    - [Key Features](#key-features)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
    - [Quick Start](#quick-start-1)
    - [Pipeline Workflow](#pipeline-workflow)
- [Script Explanation](#script-explanation)
    - [Scripts and Example Code](#scripts-and-example-code)
    - [Project File Explanation](#project-file-explanation)
- [More Information](#more-information)

## GAD-MALL Description

GAD-MALL is a deep learning framework based on Active Learning and 3D Convolutional Neural Networks (3D CNN) designed to tackle multi-objective high-dimensional optimization problems. By integrating generative models, Finite Element Method (FEM), and 3D printing technology, this framework offers an efficient data-driven design approach, particularly suitable for optimizing materials with complex structures. It is specifically applied to achieve efficient optimization of building materials, especially for complex multi-objective optimization problems, such as the application of heterogeneous materials in bioengineering and materials science. For example, it can be used to design bone graft scaffolds by optimizing the scaffold's elastic modulus and yield strength, resulting in a heterogeneous structure with biocompatibility and high mechanical strength.

### Key Features

1. **Generative Architecture Design (GAD)**: GAD uses an autoencoder network to generate a set of architectures with unknown properties. The autoencoder, through unsupervised learning, transforms the exploration of the high-dimensional design space into a low-dimensional space, effectively representing high-dimensional data and making the design process more efficient.

2. **Multi-objective Active Learning Loop (MALL)**: MALL iteratively queries the Finite Element Method (FEM) to evaluate the generated datasets, gradually optimizing the architecture's performance. This method continuously updates the training data through an active learning loop, progressively improving the accuracy of the model's predictions.

3. **3D Printing and Testing**: Architected materials designed by the ML framework are manufactured using laser powder bed fusion technology, and their mechanical properties are experimentally validated.

> Paper: Peng, B., Wei, Y., Qin, Y. et al. Machine learning-enabled constrained multi-objective design of architected materials. Nat Commun 14, 6630 (2023). https://doi.org/10.1038/s41467-023-42415-y
## Dataset

The primary datasets used in this project include the following files:

- Input Data:
    - `3D_CAE_Train.npy`: Training data for the 3D convolutional autoencoder, stored as a NumPy array.
    - `Matrix12.npy` and `Matrix60.npy`: These files contain matrix data for different - configurations used in the architecture generation and optimization process.
    - `E.csv`: A data file containing the elastic modulus of materials.
    - `yield.csv`: A data file containing the yield strength of materials.

These datasets support the training and testing of various models within the GAD-MALL framework.

- Data Download:
    - `Matrix12.npy`, `E.csv`, and `yield.csv` are located in the `./src/data` directory:

        ```txt
        ├── data
        │   ├── E.csv
        │   ├── yield.csv
        │   ├── Matrix12.npy
        │   └── README.txt
        ```

    - `3D_CAE_Train.npy` can be downloaded via the link provided in README.txt [here](https://drive.google.com/file/d/1BfmD4bsPS2hG5zm7XGLHc8lpUN_WqhgV/view?usp=share_link).
    - `Matrix60.npy can` be downloaded via the link provided in README.txt [here](https://drive.google.com/file/d/1VRH4X_mACxM82HoaplwV0ThaDiN3iPXm/view?usp=share_link).

- **Preprocessing**: Before use, the data needs to be normalized, and may require cropping or padding to fit the input size of the model.

## Environment Requirements

This project is based on the MindSpore deep learning framework. Below are the main (**test/development**) environment dependencies:

- **Hardware** (GPU)
    - GPU: NVIDIA GeForce RTX 4060 Laptop GPU
    - Driver: CUDA 12.3
    - CUDA: 11.6
    - CUDNN: 8.4.1
- * *Operating System**:
    - Windows WSL Ubuntu-2 0.04
- **Python Version**:
    - Python 3.9
- **Framework**
    - [MindSpore](https://www.mindspore.cn/install/)
- **Dependencies**:
    - mindspore==2.2.14
    - numpy==1.23.5
    - scipy==1.13.1
    - pandas==2.2.2
    - matplotlib==3.9.1
    - tqdm==4.66.5
    - You can install the dependencies using the following command:

      ```bashpython3.9 -u pip install -r requirement.txt```

- For more information, please refer to the following resources:
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/zh-CN/r2.2/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

## Quick Start

### Quick Start

After installing MindSpore from the official website, you can start training and validation as follows:

- Run on GPU

   ```bash run_GAD_MALL.sh```

### Pipeline Workflow

After installing MindSpore from the official website and downloading the required datasets, you can begin running the training and generating the Generative Architecture Design-Multi-objective Active Learning Loop (GAD-MALL) pipeline on GPU. Please follow the steps below:

  1. Train the 3D-CAE model as the generative model for GAD-MALL. Run the following command in the terminal:

      ```python3.9 3D_CAE_ms.py```

  2. Train the 3D-CNN model as the surrogate model for GAD-MALL. Run the following command in the terminal:

      ```python3.9 3D_CNN_ms.py```

  3. Use GAD-MALL to search for high-performance architected materials with specific elastic modulus and high yield strength. Run the following command in the terminal:

      ```python3.9 -u GAD_MALL_Active_learning.py```

  4. After completing the GAD-MALL process, you will obtain porosity matrices with specific predicted elastic modulus (E=2500 MPa, E=5000 MPa) and the highest predicted yield strength.

## Script Explanation

### Scripts and Example Code

The file structure is as follows:

```text

├── gad_mall
│   ├── data                          # Data files
│   │   ├── E.csv                     # Data file containing the elastic modulus of materials
│   │   ├── yield.csv                 # Data file containing the yield strength of materials
│   │   ├── Matrix12.npy              # Matrix data used for architecture generation and optimization
│   │   ├── (Matrix60.npy)            # Matrix data used for architecture generation and optimization
│   │   ├── (3D_CAE_Train.npy)        # Training data for the 3D convolutional autoencoder, stored as a NumPy array
│   │   └── README.txt                # Download links for datasets
│   ├── model                         # Directory for storing checkpoint files
│   ├── results                       # Directory for storing experimental results
│   ├── src                           # Source code
│   │   ├── 3D_CAE_ms.py              # Implementation of the 3D convolutional autoencoder
│   │   ├── 3D_CNN_ms.py              # Implementation of the 3D convolutional neural network model
│   │   └── GAD_MALL_Active_learning.py # Implementation of the GAD-MALL framework
│   ├── README.md                     # English documentation for the model
│   ├── README_CN.md                  # Chinese documentation for the model
│   ├── run_GAD_MALL.sh               # Script for starting the training process
│   └── requirements.txt              # Python environment dependencies

```

### Project File Explanation

- `3D_CNN_ms.py`: Implements a model based on 3D Convolutional Neural Networks, suitable for handling three-dimensional datasets, particularly in high-dimensional multi-objective optimization problems. This model voxelizes the input data and uses 3D convolutional layers to extract high-level features, ultimately predicting material properties.

- `GAD_MALL_Active_learning_ms.py`: Implements the active learning strategy for the GAD-MALL framework, used to optimize models in scenarios where data labeling is costly. This script combines generative models and the Finite Element Method (FEM) to iteratively search for high-performance architectures through active learning.

- `3D_CAE_ms.py`: Implements a 3D Convolutional Autoencoder, used for feature extraction or dimensionality reduction in unsupervised learning. This autoencoder is a key component in the Generative Architecture Design (GAD) process. It uses an encoder-decoder network to represent input data in a low-dimensional space and reconstruct the original data.

- `data/`: Directory containing training and testing datasets.

- `models/`: Directory for storing trained models and weight files.

- `results/`: Directory for storing the results of model inference and evaluation.

- `requirements.txt`: File listing the Python environment dependencies.

## More Information

For additional details, please refer to the original project documentation for[GAD-MALL](https://github.com/Bop2000/GAD-MALL/tree/main)