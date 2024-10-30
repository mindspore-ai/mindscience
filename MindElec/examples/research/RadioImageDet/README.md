# Radio signal recognition based on image deep learning

## Introduction

### Background

With the advent of the Internet era and the rise of the Internet of Things, radio waves have become an important carrier of information communication, but their openness also makes them vulnerable to jamming and illegal use, so the monitoring and identification of radio signals has important practical significance and demand in terms of communication security and spectrum management. Therefore, it is necessary to improve the performance of radio monitoring with the help of AI.

The RadioImageDet algorithm transforms the radio signal detection problem into an image recognition target detection problem, thereby improving the intelligence level of radio signal detection and the detection capability in complex electromagnetic environments. More information can be found in the paper,[Radio signal recognition based on image deep learning](https://xueshu.baidu.com/usercenter/paper/show?paperid=1s6y0xj0922m0mb0r11d0pc0b0712061&site=xueshu_se)

 This case tutorial describes the implementation of the RadioImageDet algorithm.

### Technical path

The technical idea of the algorithm is shown in the figure below, which is mainly divided into two parts: first, the radio signal is visualised as a two-dimensional image in the pre-processing module; then, the radio signal of the visualised two-dimensional image is detected using the target detection algorithm. This method avoids manual feature extraction based on experience and provides a new idea for the research of intelligent radio signal recognition technology in complex electromagnetic environment.

![image-20241028010742781](image/image-20241028010742781.png)

In response to the difficulty of obtaining the dataset in the original paper, this case provides a signal generation code signal_general.pyimage to simulate the original I/Q sampling data stream.
where the target detection algorithm is based on YOLOv2's RadioYOLO, the structure of which is shown below。

![image-20241028011520857](image/image-20241028011520857.png)

RadioYOLO has been improved based on YOLOv5.

## QuickStart

The dataset can be generated using the util/image_general.py script to generate image data for each type of signal, deposited in the Vocdevkit/VOC2007/JPEGImages folder.
Then use the image_img annotation tool to annotate by yourself, and deposit the annotated xml file into the Vocdevkit/VOC2007/Annotations folder.
Finally, use the util/voc_annotation.py script to split the dataset.

### YOLOv2_Based

Please first configure the parameters in train_v2.py according to the comment requirement.

|    Parameter    |                                                                                                              Setting                                                                                                               |
|:---------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|   model_path    |                                                                         Path to model file, if empty then automatically initialise parameters for training                                                                         |
| label_smoothing |                                                                                          Label smoothing, typically set to less than 0.01                                                                                          |
|  freeze_train   | Set to True to use freeze training of the backbone network, which needs to be used when a pre-trained model is available. In this case freeze the training freeze_epoch sub-epoch first, and then train the whole thing afterwards |
|   save_period   |                                                                                               Save weights every save_period epochs                                                                                                |
|    save_dir     |                                                                                          The folder where weights and log files are saved                                                                                          |
|    eval_flag    |                                                          Option to evaluate at training time for the validation set. Evaluation is performed every eval_period sub-epoch                                                           |
|   num_workers   |                                                                                             Number of threads occupied by reading data                                                                                             |

Afterward, the script is called from the command line for training.

```text
python train_v2.py
```

Before predicting, please change the parameter model_path to your model path in nets/yolo_predicting_v2.py.

Prediction with predict_v2.py script

```text
python predict_v2.py
```

### YOLOv5_Based

Please first configure the parameters in train_v5.py according to the comment requirement.

|    Parameter    |                                                                                                              Setting                                                                                                               |
|:---------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|   model_path    |                                                                         Path to model file, if empty then automatically initialise parameters for training                                                                         |
| label_smoothing |                                                                                          Label smoothing, typically set to less than 0.01                                                                                          |
|  freeze_train   | Set to True to use freeze training of the backbone network, which needs to be used when a pre-trained model is available. In this case freeze the training freeze_epoch sub-epoch first, and then train the whole thing afterwards |
|   save_period   |                                                                                               Save weights every save_period epochs                                                                                                |
|    save_dir     |                                                                                          The folder where weights and log files are saved                                                                                          |
|    eval_flag    |                                                          Option to evaluate at training time for the validation set. Evaluation is performed every eval_period sub-epoch                                                           |
|   num_workers   |                                                                                             Number of threads occupied by reading data                                                                                             |

Afterward, the script is called from the command line for training.

```text
python train_v5.py
```

Before predicting, please change the parameter model_path to your model path in nets/yolo_predicting_v5.py.

Prediction with predict_v5.py script

```text
python predict_v5.py
```

## Results Display

![image-20241028014543824](image/image-20241028014543824.png)

## Performance

|        参数         |               YOLOv2_based                |               YOLOv5_based                |
|:-----------------:|:-----------------------------------------:|:-----------------------------------------:|
|     Hardware      |                Ascend, 32G                |                Ascend, 32G                |
| MindSpore version |                  2.2.14                   |                  2.2.14                   |
|      Dataset      | image_general Generate simulation dataset | image_general Generate simulation dataset |
|     Optimizer     |                    SGD                    |                    SGD                    |
|    Train Loss     |                   0.124                   |                   0.096                   |
|    Evaluation     |                   0.119                   |                   0.103                   |

### YOLOv2_Based

![image-20241028015407296](image/image-20241028015407296.png)

### YOLOv5_Based

![image-20241028015423840](image/image-20241028015423840.png)

## Contributor

gitee id：[xiaofang666666](https://gitee.com/xiaofang666666)

email: 3290352431@qq.com