ENGLISH | [简体中文](README_CN.md)

# Contents

- [Conservative PINNs Description](#conservative-pinns-description)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
    - [Evaluation Process](#evaluation-process)

## [Conservative PINNs Description](#contents)

Proposing a conservative physics-informed neural network (cPINN) for nonlinear conservation laws on discrete domains,
this method enforces flux continuity, offers optimization freedom, and adapts activation functions for faster training.
It efficiently enables parallelized computation, solving various test cases and accommodating complex structures.

> [paper](https://www.sciencedirect.com/science/article/abs/pii/S0045782520302127): Jagtap A D, Kharazmi E, Karniadakis
> G E. Conservative physics-informed neural networks on discrete domains for conservation laws: Applications to forward
> and inverse problems[J]. Computer Methods in Applied Mechanics and Engineering, 2020, 365: 113028.

Example details: Conservative PINN code with 4 spatial subdomains for the one-dimensional Burgers equations.

## [Dataset](#contents)

The training dataset and pretrained checkpoint files will be downloaded automatically at the first launch.

Dataset used: [burgers shock]

- Dataset size
    - x: (256, 1) in [-1, 1]
    - t: (100, 1) in [0, 1]
- Data format: `.mat` files
    - Note: Data will be processed in `process.py`
- The dataset is in the `./data` directory, the directory structure is as follows:

```text
├── data
│   ├── burgers_shock.mat
│   └── L2error_Bur4SD_200Wi.mat
```

If you need to download the dataset or checkpoint files manually,
please visit [this link](https://download.mindspore.cn/mindscience/SciAI/sciai/model/cpinns/).

## [Environment Requirements](#contents)

- Hardware(Ascend/GPU)
    - Prepare hardware environment with Ascend or GPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below:
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
    --nn_depth 4 6 6 4 \
    --nn_width 20 20 20 20 \
    --save_ckpt true \
    --save_fig true \
    --load_ckpt false \
    --save_ckpt_path ./checkpoints \
    --load_ckpt_path ./checkpoints/model_final.ckpt \
    --load_data_path ./data \
    --save_data_path ./data \
    --figures_path ./figures \
    --log_path ./logs \
    --print_interval 10 \
    --ckpt_interval 1000 \
    --lr 8e-4 \
    --epochs 15001 \
    --download_data cpinns \
    --force_download false \
    --amp_level O3 \
    --device_id 0 \
    --mode 0
```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

File structures are as follows:

```text
├── cpinns
│   ├── checkpoints                  # checkpoints files
│   ├── data                         # data files
│   │   ├── burgers_shock.mat        # burgers shock matlab dataset
│   │   └── L2error_Bur4SD_200Wi.mat # result dataset for l2 error case
│   ├── figures                      # plot figures
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

| parameter      | description                                  | default value                  |
|----------------|----------------------------------------------|--------------------------------|
| nn_depth       | neural network depths                        | 4 6 6 4                        |
| nn_width       | neural network widths                        | 20 20 20 20                    |
| save_ckpt      | whether save checkpoint or not               | true                           |
| save_fig       | whether save and plot figures or not         | true                           |
| load_ckpt      | whether load checkpoint or not               | false                          |
| save_ckpt_path | checkpoint saving path                       | ./checkpoints                  |
| load_ckpt_path | checkpoint loading path                      | ./checkpoints/model_final.ckpt |
| load_data_path | path to load data                            | ./data                         |
| save_data_path | path to save data                            | ./data                         |
| figures_path   | figures saving path                          | ./figures                      |
| log_path       | log saving path                              | ./logs                         |
| print_interval | time and loss print interval                 | 10                             |
| ckpt_interval  | checkpoint save interval                     | 1000                           |
| lr             | learning rate                                | 8e-4                           |
| epochs         | number of epochs                             | 15001                          |
| download_data  | necessary dataset and/or checkpoints         | cpinns                         |
| force_download | whether download the dataset or not by force | false                          |
| amp_level      | MindSpore auto mixed precision level         | O3                             |
| device_id      | device id to set                             | None                           |
| mode           | MindSpore Graph mode(0) or Pynative mode(1)  | 0                              |

### [Training Process](#contents)

- running on GPU/Ascend

  ```bash
  python train.py
  ```

  The loss values during training will be printed in the console, which can also be inspected after training in log
  file.

  ```bash
  # grep "loss1:" log
  step: 0, loss1: 2.1404986, loss2: 8.205103, loss3: 37.23588, loss4: 3.56359, interval: 50.85803508758545s, total: 50.85803508758545s
  step: 10, loss1: 2.6560388, loss2: 3.869413, loss3: 9.323585, loss4: 2.1194165, interval: 5.159524917602539s, total: 56.01756000518799s
  step: 20, loss1: 1.7885156, loss2: 4.470225, loss3: 3.3072894, loss4: 1.5674783, interval: 1.8615927696228027s, total: 57.87915277481079s
  step: 30, loss1: 1.8574346, loss2: 3.8972874, loss3: 2.103153, loss4: 1.2108151, interval: 1.7992112636566162s, total: 59.67836403846741s
  step: 40, loss1: 1.8863815, loss2: 2.7914107, loss3: 1.4245809, loss4: 0.94769603, interval: 1.8828914165496826s, total: 61.56125545501709s
  step: 50, loss1: 1.1929171, loss2: 1.5765706, loss3: 0.758412, loss4: 0.6086196, interval: 1.6731781959533691s, total: 63.23443365097046s
  step: 60, loss1: 0.7861989, loss2: 1.5977213, loss3: 0.56675017, loss4: 0.39846048, interval: 1.6708331108093262s, total: 64.90526676177979s
  step: 70, loss1: 0.33681053, loss2: 1.0673326, loss3: 0.5887743, loss4: 0.3366256, interval: 1.8425297737121582s, total: 66.74779653549194s
  step: 80, loss1: 0.29425326, loss2: 0.9776688, loss3: 0.5781496, loss4: 0.28926677, interval: 1.829559564590454s, total: 68.5773561000824s
  step: 90, loss1: 0.16654292, loss2: 0.9878452, loss3: 0.5724378, loss4: 0.2396864, interval: 1.784325122833252s, total: 70.36168122291565s
  step: 100, loss1: 0.11878409, loss2: 0.9726932, loss3: 0.5552572, loss4: 0.21440478, interval: 1.912705659866333s, total: 72.27438688278198s
  ...
  ```

- After training, you can still review the training process through the log file saved in `log_path`, `./logs` directory
  by default.

- The model checkpoint will be saved in `save_ckpt_path`, `./checkpoint` directory by default.

### [Evaluation Process](#contents)

Before running the command below, please check the checkpoint loading path `load_ckpt_path` specified
in `config.yaml` for evaluation.

- running on GPU/Ascend

  ```bash
  python eval.py
  ```

  You can view the process and results through the `log_path`, `./logs` by default.
  The result pictures are saved in `figures_path`, [`./figures`](./figures) by default.