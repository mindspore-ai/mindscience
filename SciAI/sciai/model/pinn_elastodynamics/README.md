ENGLISH | [简体中文](README_CN.md)

# Contents

- [PINN elastodynamics Description](#pinn-elastodynamics-description)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
    - [Evaluation Process](#evaluation-process)

## [PINN elastodynamics Description](#contents)

This paper presents a physics-informed neural network (PINN) with mixed-variable output to model elastodynamics problems
without resorting to labeled data, in which the initial/boundary conditions (I/BCs) are hardly imposed.

> [paper](https://arxiv.org/abs/2006.08472): Rao C, Sun H, Liu Y. Physics-informed deep learning for computational
> elastodynamics without labeled data[J]. Journal of Engineering Mechanics, 2021, 147(8): 04021043.

Example details:

- **ElasticWaveInfinite**: Training script and dataset for elastic wave propagation in infinite domain.

## [Dataset](#contents)

The training dataset and pretrained checkpoint files will be downloaded automatically at the first launch.

Dataset used: [burgers shock]

- Dataset size
    - x: (256, 1) in [-1, 1]
    - t: (100, 1) in [0, 1]
- Data format：`.mat` files
    - Note：Data will be processed in `process.py`
- The dataset is in the `./data` directory, the directory structure is as follows:

```text
├── data
│   ├── FEM_result
│   └── burgers_shock.mat
```

If you need to download the dataset and checkpoint files manually,
please visit [this link](https://download.mindspore.cn/mindscience/SciAI/sciai/model/pinn_elastodynamics/).

## [Environment Requirements](#contents)

- Hardware(Ascend/GPU)
    - Prepare hardware environment with Ascend or GPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

## [Quick Start](#contents)

After installing MindSpore via the official website and the required [dataset](#dataset) above, you can start training
and evaluation as follows:

- running on Ascend or on GPU

Default:

```bash
python train.py
```

Full command:

```bash
python train.py \
    --uv_layers 3 140 140 140 140 140 140 7 \
    --save_ckpt true \
    --save_fig true \
    --load_ckpt false \
    --save_ckpt_path ./checkpoints \
    --load_data_path ./data \
    --load_ckpt_path ./checkpoints/uv_NN_14s_float32_new.pickle \
    --figures_path ./figures/output \
    --log_path ./logs \
    --print_interval 1 \
    --ckpt_interval 1000 \
    --lr 1e-3 \
    --epochs 100000 \
    --use_lbfgs false \
    --max_iter_lbfgs 10000 \
    --download_data pinn_elastodynamics \
    --force_download false \
    --amp_level O3 \
    --device_id 0 \
    --mode 0
```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

File structures are as follows:

```text
├── pinn_elastodynamics
│   ├── checkpoints                  # checkpoints files
│   ├── data                         # data files
│   │   ├── FEM_result               # results of the model
│   │   └── burgers_shock.mat        # burgers shock matlab dataset
│   ├── figures                      # plot figures
│   │   ├── output                   # output figures
│   │   └── GIF_uv.gif               # result animation gif
│   ├── logs                         # log files
│   ├── src                          # source codes
│   │   ├── network.py               # network architecture
│   │   ├── plot.py                  # plotting results
│   │   └── process.py               # data process
│   ├── config.yaml                  # hyper-parameters configuration
│   ├── README.md                    # English model descriptions
│   ├── README_CN.md                 # Chinese model description
│   ├── train.py                     # python training script
│   └── eval.py                      # python evaluation script
```

### [Script Parameters](#contents)

Important parameters in train.py are as follows:

| parameter      | description                                  | default value                              |
|----------------|----------------------------------------------|--------------------------------------------|
| uv_layers      | uv net layer-wise width                      | 3 140 140 140 140 140 140 7                |
| save_ckpt      | whether save checkpoint or not               | true                                       |
| save_fig       | whether to save and plot figures             | true                                       |
| load_ckpt      | whether load checkpoint or not               | false                                      |
| save_ckpt_path | checkpoint saving path                       | ./checkpoints                              |
| load_ckpt_path | checkpoint loading path                      | ./checkpoints/uv_NN_14s_float32_new.pickle |
| load_data_path | path to load data                            | ./data                                     |
| figures_path   | figures saving path                          | ./figures/output                           |
| log_path       | log saving path                              | ./logs                                     |
| print_interval | time and loss print interval                 | 1                                          |
| ckpt_interval  | checkpoint interval                          | 1000                                       |
| lr             | learning rate                                | 1e-3                                       |
| epochs         | number of epochs                             | 100000                                     |
| use_lbfgs      | whether to use L-BFGS after Adam or not      | false                                      |
| max_iter_lbfgs | maximum iteration of lbfgs                   | null                                       |
| download_data  | necessary dataset and/or checkpoints         | pinn_elastodynamics                        |
| force_download | whether download the dataset or not by force | false                                      |
| amp_level      | MindSpore auto mixed precision level         | O3                                         |
| device_id      | device id to set                             | None                                       |
| mode           | MindSpore Graph mode(0) or Pynative mode(1)  | 0                                          |

### [Training Process](#contents)

- running on GPU/Ascend

  ```bash
  python train.py
  ```

  The loss values during training will be printed in the console, which can also be inspected after training in log
  file.

  ```bash
  # grep "loss:" log
  step: 0, loss: 13.111078, interval: 34.97392821311951s, total: 34.97392821311951s
  step: 1, loss: 200.82831, interval: 0.08405542373657227s, total: 35.05798363685608s
  step: 2, loss: 25.90921, interval: 0.0743703842163086s, total: 35.13235402107239s
  step: 3, loss: 14.451968, interval: 0.08811116218566895s, total: 35.22046518325806s
  step: 4, loss: 48.904766, interval: 0.0770270824432373s, total: 35.297492265701294s
  step: 5, loss: 34.188297, interval: 0.08126688003540039s, total: 35.378759145736694s
  step: 6, loss: 6.9077187, interval: 0.07222247123718262s, total: 35.45098161697388s
  step: 7, loss: 3.6523025, interval: 0.07272839546203613s, total: 35.52371001243591s
  step: 8, loss: 17.293848, interval: 0.0725715160369873s, total: 35.5962815284729s
  step: 9, loss: 20.209349, interval: 0.0718080997467041s, total: 35.668089628219604s
  step: 10, loss: 11.190135, interval: 0.07178211212158203s, total: 35.73987174034119s
  ...
  ```

- After training, you can still review the training process through the log file saved in `log_path`, `./logs` directory
  by default.

- The model checkpoint will be saved in `save_ckpt_path`, `./checkpoint` directory by default.

### [Evaluation Process](#contents)

Before running the command below, please check the checkpoint path used for evaluation.

- running on GPU/Ascend

  ```bash
  python eval.py
  ```

You can view the process and results through the `log_path`, `./logs` by default.
The result pictures are saved in `figures_path`, [`./figures`](./figures) by default.