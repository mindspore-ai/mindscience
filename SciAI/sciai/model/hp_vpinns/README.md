# Contents

- [hp-VPINNs Description](#hp-vpinns-description)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
    - [Evaluation Process](#evaluation-process)

## [hp-VPINNs Description](#contents)

The study proposes a framework for hp-variational physics-informed neural networks (hp-VPINNs) that uses a global neural
network trial space and piecewise polynomial test space with hp-refinement via domain decomposition and projection onto
high-order polynomials.
This approach improves accuracy and reduces training cost for function approximation and solving differential equations.

> [paper](https://arxiv.org/abs/2003.05385): Kharazmi, Ehsan, Zhongqiang Zhang, and George E. Karniadakis. hp-VPINNs:
> Variational Physics-Informed Neural Networks With Domain Decomposition, arXiv preprint arXiv:2003.05385 (2020).

## [Dataset](#contents)

The dataset is generated during runtime.
The size of dataset is controlled by number of quadrature points `n_quad` and number of samples for each factor `n_f`
in `config.yaml`, and by default are 80 and 500, respectively.

The pretrained checkpoint files will be downloaded automatically at the first launch.
If you need to download the checkpoint files manually,
please visit [this link](https://download.mindspore.cn/mindscience/SciAI/sciai/model/hp_vpinns/).

## [Environment Requirements](#contents)

- Hardware(Ascend/GPU)
    - Prepare hardware environment with Ascend or GPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below:
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

## [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

- running on Ascend or on GPU

Default:

```bash
python train.py
```

Full command:

```bash
python train.py \
    --layers 1 20 20 20 20 1 \
    --save_ckpt true \
    --load_ckpt false \
    --save_fig true \
    --save_ckpt_path ./checkpoints \
    --load_ckpt_path ./checkpoints/model_40001.ckpt \
    --figures_path ./figures \
    --log_path ./logs \
    --lr 1e-3 \
    --epochs 40001 \
    --early_stop_loss 2e-32 \
    --var_form 1 \
    --n_element 4 \
    --n_testfcn 60 \
    --n_quad 80 \
    --n_f 500 \
    --lossb_weight 1 \
    --font 24 \
    --download_data hp_vpinns \
    --force_download false \
    --amp_level O3 \
    --device_id 0 \
    --mode 0
```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

File structures are as follows:

```text
├── hp_vpinns
│   ├── checkpoints          # checkpoints files
│   ├── data                 # data files
│   ├── figures              # plot figures
│   ├── logs                 # log files
│   ├── src                  # source codes
│   │   ├── network.py       # network architecture
│   │   ├── plot.py          # plotting results
│   │   └── process.py       # data process
│   ├── config.yaml          # hyper-parameters configuration
│   ├── README.md            # English model descriptions
│   ├── README_CN.md         # Chinese model description
│   ├── train.py             # python training script
│   └── eval.py              # python evaluation script
```

### [Script Parameters](#contents)

Important parameters in train.py are as follows:

| parameter       | description                                                                                   | default value                  |
|-----------------|-----------------------------------------------------------------------------------------------|--------------------------------|
| layers          | layer-wise width                                                                              | 1 20 20 20 20 1                |
| save_ckpt       | whether save checkpoint or not                                                                | true                           |
| save_fig        | whether to save and plot figures                                                              | true                           |
| load_ckpt       | whether load checkpoint or not                                                                | false                          |
| save_ckpt_path  | checkpoint saving path                                                                        | ./checkpoints                  |
| load_ckpt_path  | checkpoint loading path                                                                       | ./checkpoints/model_40001.ckpt |
| figures_path    | figures saving path                                                                           | ./figures                      |
| log_path        | log saving path                                                                               | ./logs                         |
| lr              | learning rate                                                                                 | 1e-3                           |
| epochs          | number of epochs                                                                              | 40001                          |
| early_stop_loss | early stop threshold for loss                                                                 | 2e-32                          |
| var_form        | variational form                                                                              | 1                              |
| n_element       | number of elements                                                                            | 4                              |
| n_testfcn       | number of points for test function                                                            | 60                             |
| n_quad          | number of quadrature points                                                                   | 80                             |
| n_f             | number of samples for each factor                                                             | 500                            |
| lossb_weight    | weight factor for lossb                                                                       | 1                              |
| font            | font size for plotting                                                                        | 24                             |
| download_data   | necessary dataset and/or checkpoints                                                          | hp_vpinns                      |
| force_download  | whether download the dataset or not by force                                                  | false                          |
| amp_level       | MindSpore auto mixed precision level                                                          | O3                             |
| device_id       | device id to set                                                                              | None                           |
| mode            | MindSpore Graph mode(0) or Pynative mode(1). This model currently doesn't support Graph mode. | 0                              |

### [Training Process](#contents)

- running on GPU/Ascend

  ```bash
  python train.py
  ```

  The loss values during training will be printed in the console, which can also be inspected after training in log
  file.

  ```bash
  # grep "loss:" log
  step: 0, loss: 142.92845, lossb: 0.7792306, lossv: 142.14922, interval: 9.788227081298828s, total: 9.788227081298828s
  step: 10, loss: 142.28386, lossb: 0.15564875, lossv: 142.1282, interval: 34.56686305999756s, total: 44.35509014129639s
  step: 20, loss: 142.10016, lossb: 0.003990522, lossv: 142.09616, interval: 34.076114892959595s, total: 78.43120503425598s
  step: 30, loss: 142.12141, lossb: 0.04219491, lossv: 142.07922, interval: 34.79131770133972s, total: 113.2225227355957s
  step: 40, loss: 142.09048, lossb: 0.004216266, lossv: 142.08627, interval: 34.17010521888733s, total: 147.39262795448303s
  step: 50, loss: 142.08542, lossb: 5.215431e-05, lossv: 142.08537, interval: 34.377405881881714s, total: 181.77003383636475s
  step: 60, loss: 142.07175, lossb: 0.0014050857, lossv: 142.07034, interval: 34.402318477630615s, total: 216.17235231399536s
  step: 70, loss: 142.05489, lossb: 0.0055830916, lossv: 142.0493, interval: 34.186588525772095s, total: 250.35894083976746s
  step: 80, loss: 142.02605, lossb: 0.0036653157, lossv: 142.02238, interval: 34.496164321899414s, total: 284.85510516166687s
  step: 90, loss: 141.97115, lossb: 0.0032817253, lossv: 141.96786, interval: 34.388723611831665s, total: 319.24382877349854s
  step: 100, loss: 141.83458, lossb: 0.0045128292, lossv: 141.83006, interval: 34.6738965511322s, total: 353.91772532463074s
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