
# 模型名称

> DeepDFT

## 介绍

> Equivariant DeepDFT模型基于等变消息传递神经网络，能够处理旋转等变性的问题。模型通过在图中插入特殊的探测节点来计算密度，与其他模型不同，该方法仅需要原子序数和原子坐标作为输入，不依赖于预定义的基函数集。

## 数据集

> 支持下面三个数据集，建议选择NMC数据集，仅为10GB左右。下载后将数据文件放置在`data/nmc/`目录下。

* QM9 Charge Densities and Energies Calculated with VASP [[link]](https://data.dtu.dk/articles/dataset/QM9_Charge_Densities_and_Energies_Calculated_with_VASP/16794500)
* NMC Li-ion Battery Cathode Energies and Charge Densities [[link]](https://data.dtu.dk/articles/dataset/NMC_Li-ion_Battery_Cathode_Energies_and_Charge_Densities/16837721)
* Ethylene Carbonate Molecular  [[link]](https://data.dtu.dk/articles/dataset/Ethylene_Carbonate_Molecular_Dynamics/16691825)

## 环境要求

> 1. 安装`mindspore（2.3.1）`
> 2. 安装依赖包：`pip install -r requirement.txt`

## 快速入门

> 1. 下载数据集到当前目录
> 2. 运行训练命令：`python train.py`
> 3. 运行测试命令：`python evaluate_model.py`
> 4. 运行推理命令：`python predict_with_model.py`

### 代码目录结构

```txt
DeepDFT
    │  README.md    		 README文件
    │  train.py    			 训练脚本
    │  requirement.txt    	 环境依赖
	│  evaluate_model.py	 测试脚本
	│  predict_with_model.py 推理脚本
    └─ config.yaml    配置文件
            config.yaml      训练过程配置文件，包含模型数据性质等参数
            config_eval.yaml 测试过程配置文件
            config_pred.yaml 推理过程配置文件
    └─ data           数据集
            nmc              NMC数据列表和分割方式文件
            qm9              QM9数据列表和分割方式文件
            predict          用于推理的葡萄糖分子案例数据
    └─ src
            dataset.py       用于读取数据并进行预处理
            densitymodel.py  密度模型：原子表示模型和消息传递网络
            layer.py         模型中部分算子实现
            utils.py         电荷密度写入文件接口
    └─ checkpoints    三个数据集对应的模型检查点
            ethylenecarbonate_painn
            nmc_painn
            qm9_painn
```

## 训练推理过程

### 训练

```bash
pip install -r requirement.txt
python train.py --config configs/config.yaml
```

训练结果日志：
```log
2024-11-05 09:10:57,103 [INFO ]  loading data ../data/NMC/nmc.txt
2024-11-05 09:10:57,407 [INFO ]  train size: 1450, val size: 50
2024-11-05 09:32:38,284 [INFO ]  Preloading training batch
2024-11-05 09:33:36,216 [INFO ]  Preloading validation batch
2024-11-05 09:33:41,020 [DEBUG]  model has 1487233 parameters
2024-11-05 09:35:45,319 [INFO ]  step=0, val_mae=0.776491, val_rmse=1.47341, sqrt(train_loss)=1.46223
2024-11-05 09:35:45,546 [DEBUG]  data_timer 0.642158 (0.642158) transfer_timer 0.000001 (0.000001) train_timer 120.108392 (120.108392) eval_time 3.498260 (3.498260)
2024-11-05 10:03:34,766 [INFO ]  step=5000, val_mae=0.010186, val_rmse=0.0154951, sqrt(train_loss)=0.178219
2024-11-05 10:03:34,999 [DEBUG]  data_timer 0.001794 (0.006819) transfer_timer 0.000001 (0.000001) train_timer 0.286871 (0.350138) eval_time 3.352099 (3.425180)
...
2024-11-19 22:52:51,507 [INFO ]  step=3985000, val_mae=0.00062381, val_rmse=0.00143781, sqrt(train_loss)=0.000447514
2024-11-19 22:52:51,508 [DEBUG]  data_timer 0.001687 (0.005780) transfer_timer 0.000001 (0.000001) train_timer 0.323905 (0.308916) eval_time 2.696724 (2.814860)
2024-11-19 23:19:35,235 [INFO ]  step=3990000, val_mae=0.000610933, val_rmse=0.00141989, sqrt(train_loss)=0.000447992
2024-11-19 23:19:35,235 [DEBUG]  data_timer 0.001690 (0.005779) transfer_timer 0.000000 (0.000001) train_timer 0.314175 (0.308923) eval_time 2.686261 (2.814699)
```

### 评估

```bash
python evaluate_model.py --config configs/config_eval.yaml
```

评估结果日志：
```log
2024-10-28 20:48:48,197 [INFO ]  loading model from ./pretrained_models/nmc_painn/best_model.ckpt
2024-10-28 20:48:48,260 [INFO ]  loading data ../data/NMC/nmc.txt
2024-10-28 20:49:57,181 [INFO ]  split=train, filename=0.CHGCAR.lz4, mae=0.000368, rmse=0.000616, abs_relative_error=0.052451%
2024-10-28 20:50:59,495 [INFO ]  split=train, filename=2.CHGCAR.lz4, mae=0.000426, rmse=0.000703, abs_relative_error=0.063143%
2024-10-28 20:52:01,032 [INFO ]  split=train, filename=4.CHGCAR.lz4, mae=0.000387, rmse=0.000627, abs_relative_error=0.056302%
2024-10-28 20:52:57,020 [INFO ]  split=train, filename=5.CHGCAR.lz4, mae=0.000535, rmse=0.000920, abs_relative_error=0.080672%
...
```

### 推理

```bash
python evaluate_model.py --config configs/config_eval.yaml
```
