# ORB模型并行训练说明文档

本文档说明了ORB模型从单卡训练到多卡并行训练的实现方案、启动方式以及性能提升结果。

## 一、并行实现

对比`finetune.py`和`finetune_parallel.py`，主要有以下几处改动：

1、引入并行训练所需的mindspore通信模块：

```python
from mindspore.communication import init
from mindspore.communication import get_rank, get_group_size
```

2、训练步骤中增加梯度聚合：

```python
# 单卡版本
grad_fn = ms.value_and_grad(model.loss, None, optimizer.parameters, has_aux=True)

# 并行版本
grad_fn = ms.value_and_grad(model.loss, None, optimizer.parameters, has_aux=True)
grad_reducer = nn.DistributedGradReducer(optimizer.parameters)  # 新增梯度规约器
```

3、数据加载时实现数据分片：

```python
# 单卡版本
dataloader = [base.batch_graphs([dataset[j] for j in range(i, min(i + batch_size, len(dataset)))])
             for i in range(0, len(dataset), batch_size)]

# 并行版本
rank_id = get_rank()
rank_size = get_group_size()
dataloader = [[dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
             for i in range(0, len(dataset), batch_size)]
dataloader = [base.batch_graphs(
    data[rank_id*len(data)//rank_size : (rank_id+1)*len(data)//rank_size]
) for data in dataloader]
```

4、初始化并行训练环境：

```python
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True)
init()
```

## 二、启动方式

设置训练参数

> 1. 修改`configs/config_parallel.yaml`中的参数：
> a. 设置`data_path`字段指定训练和测试数据集
> b. 设置`checkpoint_path`指定预训练模型权重路径
> c. 根据需要调整其他训练参数
> 2. 修改`run_parallel.sh`中的并行数：
> a. 通过`--worker_num=4 --local_worker_num=4`设置使用卡的数量

启动训练

```bash
pip install -r requirement.txt
bash run_parallel.sh
```

## 三、性能提升

单卡训练结果如下所示：

```log
Loading datasets: dataset/train_mptrj_ase.dbTotal train dataset size: 800 samples
Loading datasets: dataset/val_mptrj_ase.dbTotal train dataset size: 200 samples
Model has 25213610 trainable parameters.
Epoch: 0/100,
 train_metrics: {'data_time': 0.00010895108183224995, 'train_time': 386.58018293464556, 'energy_reference_mae': 5.598883946736653, 'energy_mae': 3.3611322244008384, 'energy_mae_raw': 103.14391835530598, 'stress_mae': 41.36046473185221, 'stress_mae_raw': 12.710869789123535, 'node_mae': 0.02808943825463454, 'node_mae_raw': 0.0228044210622708, 'node_cosine_sim': 0.7026202281316122, 'fwt_0.03': 0.23958333333333334, 'loss': 44.74968592325846}
 val_metrics: {'energy_reference_mae': 5.316623687744141, 'energy_mae': 3.594848871231079, 'energy_mae_raw': 101.00129699707031, 'stress_mae': 30.630516052246094, 'stress_mae_raw': 9.707925796508789, 'node_mae': 0.017718862742185593, 'node_mae_raw': 0.014386476017534733, 'node_cosine_sim': 0.5506304502487183, 'fwt_0.03': 0.375, 'loss': 34.24308395385742}

...

Epoch: 99/100,
 train_metrics: {'data_time': 7.802306208759546e-05, 'train_time': 59.67856075416785, 'energy_reference_mae': 5.5912095705668134, 'energy_mae': 0.007512244085470836, 'energy_mae_raw': 0.21813046435515085, 'stress_mae': 0.7020445863405863, 'stress_mae_raw': 2.222463607788086, 'node_mae': 0.04725319395462672, 'node_mae_raw': 0.042800972859064736, 'node_cosine_sim': 0.3720853428045909, 'fwt_0.03': 0.09895833333333333, 'loss': 0.7568100094795227}
 val_metrics: {'energy_reference_mae': 5.308632850646973, 'energy_mae': 0.27756747603416443, 'energy_mae_raw': 3.251189708709717, 'stress_mae': 2.8720269203186035, 'stress_mae_raw': 9.094478607177734, 'node_mae': 0.05565642938017845, 'node_mae_raw': 0.05041291564702988, 'node_cosine_sim': 0.212838813662529, 'fwt_0.03': 0.19499999284744263, 'loss': 3.2052507400512695}
Checkpoint saved to orb_ckpts/
Training time: 7333.08717 seconds

```

四卡并行训练结果如下所示：

```log
Loading datasets: dataset/train_mptrj_ase.dbTotal train dataset size: 800 samples
Loading datasets: dataset/train_mptrj_ase.dbTotal train dataset size: 800 samples
Loading datasets: dataset/train_mptrj_ase.dbTotal train dataset size: 800 samples
Loading datasets: dataset/train_mptrj_ase.dbTotal train dataset size: 800 samples
Loading datasets: dataset/val_mptrj_ase.dbTotal train dataset size: 200 samples
Loading datasets: dataset/val_mptrj_ase.dbTotal train dataset size: 200 samples
Loading datasets: dataset/val_mptrj_ase.dbTotal train dataset size: 200 samples
Loading datasets: dataset/val_mptrj_ase.dbTotal train dataset size: 200 samples
Model has 25213607 trainable parameters.
Model has 25213607 trainable parameters.
Model has 25213607 trainable parameters.
Model has 25213607 trainable parameters.

...

Training time: 2375.89474 seconds
Training time: 2377.02413 seconds
Training time: 2377.22778 seconds
Training time: 2376.63176 seconds

```

在相同的训练配置下，并行训练相比单卡训练取得了显著的性能提升：

- 单卡训练耗时：7293.28995 seconds
- 4卡并行训练耗时：2377.22778 seconds
- 性能提升：67.40%
- 加速比：3.07倍
