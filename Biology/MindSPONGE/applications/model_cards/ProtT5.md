# ProtT5

## 模型介绍

计算生物学和生物信息学从蛋白质序列中获得了大量的数据，非常适合使用自然语言处理中的语言模型。这些语言模型以低推理成本达到了新的预测效果。ProtTrans提供了先进的预训练模型用于蛋白质研究。其中，ProtT5是项目中多个预训练模型中，效果最好的。
详细信息见项目主页及论文：[github](https://github.com/agemagician/ProtTrans)

我们提供了Mindspore框架下, ProtT5模型的checkpoint, 预测接口, 训练接口；同时提供了ProtT5预训练模型在相关下游任务中的训练接口和预测接口。
ProtTrans论文中有两类下游任务，预测蛋白质相关性质和氨基酸的性质，对应的分别是sample level和token level的分类模型; 实验具体信息可以参考作者论文和项目主页中的描述。
下游任务实验的训练数据， 评测数据都可以从项目主页中提供的链接下载。

### 模型权重获取

模型权重可以从mindspore默认的[checkpoint](https://download-mindspore.osinfra.cn/mindscience/mindsponge/ProtT5/checkpoint/)仓下载，也可以下载官方的torch权重文件转换。
torch版本的权重文件下载链接：[模型页面](https://huggingface.co/Rostlab/prot_t5_xl_uniref50)，然后使用`prot_t5/scripts`文件夹中的`convert_weight.py`脚本转换为mindspore支持的格式, 脚本使用方式如下:

```shell
python scripts/convert_weight.py --layers 24 --torch_path pytorch_model.bin --mindspore_path ./mindspore_t5.ckpt
```

转换完成后，需要添加yaml格式的配置文件; 具体可以参考： `model_configs/ProtT5/t5_xl.yaml`; 直接把这个文件复制到相应的目录中也可以; 配置文件参数的含义可以参考mindformers中t5_config定义。 这些文件在`MindSPONGE`项目下。

- 文件结构

```bash
# checkpoint文件组织格式如下
└── prot_t5_xl_uniref50
    ├── prot_t5_xl_uniref50.ckpt   # 权重文件; 需要从torch的bin文件转换而来
    ├── prot_t5.yaml               # 网络配置文件，需要手动添加
    ├── special_tokens_map.json    # tokenizer
    ├── spiece.model               # tokenizer model
    └── tokenizer_config.json      # tokenizer config
```

## 如何使用

### Dependencies

```bash
mindspore >= 2.3.0
mindformers >= 1.2.0
sentencepiece >= 0.2.0
```

### ProtT5预测

```bash
from mindsponge import PipeLine

config_path = 'configs/t5_predict.yaml'  # 根据需要改为本地路径
pipe = PipeLine(name = "ProtT5")
pipe.set_device_id(0)
pipe.initialize(config_path=config_path)

# pridict
data = ["A E T C Z A O", "S K T Z P"]
res = pipe.predict(data, mode="generate")
print("Generated:", res)
# Generated: ['A E T C X A X', 'S K T X P']

res = pipe.predict(data, mode="embedding")
print("Embedding:", res)
# Embedding:
[[[ 1.71719193e-01 -1.40796244e-01 -2.04709724e-01 ...  1.45269990e-01
    1.47509247e-01 -7.32109100e-02]
  [ 9.36630294e-02 -1.16918117e-01 -2.99756974e-01 ...  1.00125663e-01
   -2.26259604e-01  2.25636318e-01]
  [ 1.93479404e-01 -9.52076018e-02 -2.92140573e-01 ...  6.69623986e-02
    3.05505600e-02  1.31701231e-01]
    ...
```

### ProtT5预训练

```bash
# 单卡; 按照配置文件配置好yaml文件
config_path = 'configs/t5_pretrain.yaml'  # 根据需要改为本地路径
pipe = PipeLine(name = "ProtT5")
pipe.initialize(config_path=config_path)
pipe.model.init_trainer()
pipe.model.train()

# 使用多卡并行; run_pretrain.py中代码就是单卡的代码，使用msrun启动
msrun --worker_num=${worknum} --local_worker_num=${worknum} --master_port=8128 --log_dir=msrun_log --join=True --cluster_time_out=600 ./run_pretrain.py
```

### 下游任务

```bash
import mindspore as ms
from mindsponge import PipeLine

pipe = PipeLine(name = "ProtT5Downstream")
pipe.set_device_id(0)
config_path = 'configs/t5_downstream_task_eval.yaml'
pipe.initialize(config_path=config_path)

# pridict
data = ["S L R F T A S T S T P K S G S K I A K R G K K H P E P V A S W M S E Q R W A G E P E V M C T L Q H K S I A Q E A Y K N Y T I T T S A V C K L V R Q L Q Q Q A L S L Q V H F E R S E R V L S G L Q A S S L P E A L A G A T Q L L S H L D D F T A T L E R R G V F F N D A K I E R R R Y E Q H L E Q I R T V S K D T R Y S L E R Q H Y I N L E S L L D D V Q L L K R H T L I T L R L I F E R L V R V L V I S I E Q S Q C D L L L R A N I N M V A T L M N I D Y D G F R S L S D A F V Q N E A V R T L L V V V L D H K Q S S V R A L A L R A L A T L C C A P Q A I N Q L G S C G G I E I V R D I L Q V E S A G E R G A I E R R E A V S L L A Q I T A A W H G S E H R V P G L R D C A E S L V A G L A A L L Q P E"]
res = pipe.predict(data)
print("Output:", res)
# Output: ['Cytoplasm']

# 评估测试集; 项目主页有数据集下载地址
eval_data_path = "./dataset/deeploc_test_set.csv"  
pipe.model.eval_acc(eval_data_path)
# Accuracy 0.8129

# train
# config文件中设置好train_data_path和eval_data_path等参数
# yaml文件中parallel设为True
ms.set_context(mode=ms.GRAPH_MODE, device_target='Ascend', device_id=0)
pipe = PipeLine(name = "ProtT5Downstream")
config_path = 'configs/t5_downstream_task_train.yaml'
pipe.initialize(config_path=config_path)
pipe.model.train()
```

### 预训练说明

ProtTrans主要工作是在蛋白质氨基酸序列上训练的预训练模型, 下面是模型训练相关一些说明。

- 数据转换

为了训练效率，首先需要原始数据转换成mindrecord格式。原始的预训练数据可以使用`uniref50`数据, 下面是数据转换脚本的路径及其使用方式。`number_samples`指定了想转换的样本数量，默认是`-1`转换全部数据。

```shell
# 参数分别是： 原始csv数据目录; 转换后的目录; 模型checkpoint路径; 转换样本数
python scripts/trans_csv_to_mindrecord.py --data_dir ../unif50 --output_dir ../unif50_mindrecord  --t5_config_path  ../prot_t5_xl_uniref50  --number_samples  50000
```

- T5参数配置

除了网络中每层的参数量和dropout比例，下面几个参数也需要注意

```yaml
# 初始化权重缩放比例; 一般小于等于1
initializer_factor: 1.0

# 每一层的数据类型，兼容混合精度: float32 或 float16
# T5模型中建议全部使用float32
param_init_type: "float32"
layernorm_compute_type: "float32"
softmax_compute_type: "float32"
compute_dtype: "float32"
```

## 引用

```bash
@article{9477085,
  author={Elnaggar, Ahmed and Heinzinger, Michael and Dallago, Christian and Rehawi, Ghalia and Yu, Wang and Jones, Llion and Gibbs, Tom and Feher, Tamas and Angerer, Christoph and Steinegger, Martin and Bhowmik, Debsindhu and Rost, Burkhard},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  title={ProtTrans: Towards Cracking the Language of Lives Code Through Self-Supervised Deep Learning and High Performance Computing},
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TPAMI.2021.3095381}
}
```