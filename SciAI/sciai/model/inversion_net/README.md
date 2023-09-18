ENGLISH | [简体中文](README_CN.md)

# Contents

- [Inversion Net Description](#inversion-net-description)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
    - [Evaluation Process](#evaluation-process)
- [References](#References)

## [Inversion Net Description](#contents)

In order to solve the problems of high computational cost and low resolution in full waveform inversion (FWI) of seismic data, Wu and Lin proposed a novel deep neural network -- InversionNet^[1]^ in 2020. This network is a convolutional neural network (CNN) with an encoder-decoder structure, which can extract high-level features from the input seismic data, and then convert these features into a velocity model. The encoder employs convolutional blocks to extract high-level features of the input seismic data and compress them into a single high-dimensional vector. The decoder employs a hybrid convolution and deconvolution block to transform features into a velocity model.

In numerical experiments on synthetic data FlatVel and CurvedVel, InversionNet can quickly predict the velocity structure in the subsurface. And compared to traditional algorithms, InversionNet significantly improves the accuracy of inversion results while reducing computational complexity.

<center class="half">
    <img title="Logo" src="./figures/InversionNet.png"  ><br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;    display: inline-block;    color: #999;    padding: 2px;">the architecture of InversionNet</div> </center>

> [Paper](https://ieeexplore.ieee.org/document/8918045#:~:text=InversionNet:%20An%20Efficient%20and%20Accurate%20Data-Driven%20Full%20Waveform,maps%20the%20subsurface%20velocity%20structures%20to%20seismic%20signals.):
> Wu, Yue, and Youzuo Lin.
> "InversionNet: An efficient and accurate data-driven full waveform inversion."
> IEEE Transactions on Computational Imaging 6 (2019): 419-433.

## [Dataset](#contents)

Two datasets are used in this paper: FlatVel and CurvedVel. FlatVel is a velocity model that simulates a flat layer, and CurvedVel is a velocity model that simulates a curved subsurface. Here we use datasets ([FlatVel-A](https://openfwi-lanl.github.io/docs/data.html) and [CurveVel-A](https://openfwi-lanl.github.io/docs/data.html) ) provided by OpenFWI , where FlatVel is an example as shown in the figure below. In addition, OpenFWI also provides a variety of data sets, including **Vel family**, **Fault family**, and **Style family**.
<center class="half"><img title="Logo" src="./figures/Velocity.png" width="300"><img src="./figures/Seismic.png" width="610"><br><div style="color:orange; border-bottom: 1px solid #d9d9d9;display:inline-block; color: #999; padding: 2px;">Velocity model and simulated seismic data<br></div> </center>

## [Environment Requirements](#contents)

- Hardware(Ascend/GPU)
    - Prepare hardware environment with Ascend or GPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/)
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
    --case flatvel-a
    --anno_path split_files
    --train_anno flatvel_a_train.txt
    --val_anno flatvel_a_val.txt
    --print_freq 100 \
    --epoch_block 40 \
    --num_block 3 \
    --save_fig true \
    --save_ckpt true \
    --load_ckpt false \
    --save_ckpt_path ./checkpoints \
    --load_ckpt_path ./checkpoints/your_file.ckpt \
    --figures_path ./figures \
    --lr 1e-4 \
    --data_type float32
```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```text
├── inversion_net
│   ├── data                       # data files
│   │   └── README.md              # download address of datasets
│   ├── figures                    # plot figures
│   ├── src                        # source codes
│   │   ├── network.py             # network architecture
│   │   ├── plot.py                # plotting results
│   │   ├── process.py             # data process
│   │   ├── scheduler.py           # scheduler for learning rate
│   │   ├── ssim.py                # structure similarity
│   │   └── utils.py               # utils
│   ├── config.json                # hyper-parameters configuration
│   ├── README.md                  # English model descriptions
│   ├── README_CN.md               # Chinese model description
│   ├── train.py                   # python training script
│   └── eval.py                    # python evaluation script
```

### [Script Parameters](#contents)

Important parameters in `train.py` are as follows:

| parameter      | description                                  | default value                      |
| -------------- | -------------------------------------------- | ---------------------------------- |
| case           | name of the used dataset                     | flatvel-a (flatvel-a / curvevel-a) |
| anno_path      | directory of annotation file                 | ./data                             |
| train_anno     | annotation file for training data            | flatvel_a_train.txt                |
| val_anno       | annotation file for validation data          | flatvel_a_val.txt                  |
| device_target  | target device                                | GPU (CPU, GPU, Ascend )            |
| device_num     | ID of the target device                      | 0                                  |
| mode           | Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) | 0                                  |
| amp_level      | level of auto mixed precision                | O0 ( "O0", "O1", "O2", "O3")       |
| dims           | neural network widths                        | [32, 64, 128, 256, 512]            |
| lambda_g1v     | weight of L1 loss                            | 1.0                                |
| lambda_g2v     | weight of L2 loss                            | 1.0                                |
| batch_size     | batch size                                   | 128                                |
| lr             | learning rate                                | 1e-4                               |
| start_epoch    | start epoch                                  | 0                                  |
| epoch_block    | epochs in a saved block                      | 40                                 |
| num_block      | number of saved block                        | 3                                  |
| print_freq     | interval for loss printing                   | 50                                 |
| save_fig       | whether save and plot figures or not         | true                               |
| save_ckpt      | whether save checkpoint or not               | true                               |
| load_ckpt      | whether load checkpoint or not               | false                              |
| save_ckpt_path | checkpoint saving path                       | ./checkpoints                      |
| load_ckpt_path | checkpoint loading path                      | ./checkpoints/your_file.ckpt       |
| vis_path       | figures saving path                          | ./figures                          |

### [Training Process](#contents)

- Prepare Data

1. download dataset

Download the FlatVel-A or CurveVel-A dataset from [OpenFWI Benchmarks](https://openfwi-lanl.github.io/docs/data.html).

2. Prepare training and testing set

A convenient way of loading the data is to use a `.txt` file containing the _location+filename_ of all`.npy`files. Take **flatvel-A** as an example, we create`flatvel-a-train.txt`, organized as the follows, and same for `flatvel-a-test.txt`.

   ```bash
   Dataset_directory/data1.npy  Dataset_directory/model1.npy
   Dataset_directory/data2.npy  Dataset_directory/model2.npy
   Dataset_directory/data3.npy  Dataset_directory/model3.npy
   Dataset_directory/data4.npy  Dataset_directory/model4.npy
   ...
   ```

**To compare performance, we advise to download the `.txt` file from the `split_files` folder in [OpenFWI](https://github.com/lanl/OpenFWI), and then change the data path in the`.txt` file to  your own directory.**

- running on GPU/Ascend

  ```bash
  python train.py
  ```

The loss values during training will be printed in the console.

  ```bash
  mindspore version:  2.0.0
  Loading data
  Loading validation data
  Loading training data
  Epoch: [0]  [  0/187]  samples/s: 19.009  loss: 1.3781 (1.3781)  loss_g1v: 0.6870 (0.6870)  loss_g2v: 0.6911 (0.6911)  time: 7.5583  data: 0.8247
  Epoch: [0]  [ 50/187]  samples/s: 258.307  loss: 0.2868 (0.4019)  loss_g1v: 0.2039 (0.2623)  loss_g2v: 0.0847 (0.1396)  time: 0.7549  data: 0.2765
  Epoch: [0]  [100/187]  samples/s: 265.484  loss: 0.2583 (0.3328)  loss_g1v: 0.1835 (0.2248)  loss_g2v: 0.0745 (0.1080)  time: 0.8730  data: 0.3957
  Epoch: [0]  [150/187]  samples/s: 266.433  loss: 0.2463 (0.3051)  loss_g1v: 0.1750 (0.2090)  loss_g2v: 0.0711 (0.0961)  time: 0.5796  data: 0.1002
  Epoch: [0] Total time: 0:02:12
  Test:  [   0/46]  loss: 0.2594 (0.2594)  loss_g1v: 0.1849 (0.1849)  loss_g2v: 0.0745 (0.0745)  time: 2.3954  data: 0.5047
  Test:  [  20/46]  loss: 0.2596 (0.2597)  loss_g1v: 0.1844 (0.1846)  loss_g2v: 0.0752 (0.0751)  time: 0.6590  data: 0.4941
  Test:  [  40/46]  loss: 0.2612 (0.2609)  loss_g1v: 0.1853 (0.1852)  loss_g2v: 0.0760 (0.0757)  time: 0.7450  data: 0.5813
  Test: Total time: 0:00:34
   * Loss 0.26126102
  ```

- The model checkpoint will be saved in `save_ckpt_path`, `./checkpoint` directory by default.

### [Evaluation Process](#contents)

Before running the command below, please check the checkpoint loading path `load_ckpt_path` specified
in `config.json` for evaluation.

- running on GPU/Ascend

  ```bash
  python eval.py --case flatvel-a --load_ckpt_path ./checkpoints/model.ckpt --val_anno flatvel_a_val.txt
  ```

The result pictures are saved in `figures_path`, `./figures` by default.

<center class="half">
    <img title="Logo" src="./figures/Prediction.png"  ><br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;    display: inline-block;    color: #999;    padding: 2px;">A prediction result of InversionNet</div> </center>
