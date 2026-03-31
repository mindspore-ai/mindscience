# ReactGen

## 背景简介

化学反应生成基础模型以Qwen2.5-1.5B为主干模型架构，并使用训练的多模态对齐模块ReactionBERT作为SMILES输入的编码器，经过proj层投影后，与自然语言的embedding融合，再通过Decoder解码输出SMILES表达式。模型架构如下：
```
SMILES ──▶ ReactionBERT ──┐ 
                           │ proj
NLP ──▶ Qwen2.5 embedding ┤────► Qwen2.5 Decoder 
```

模型经过训练后，可用于化学反应生成领域的下游任务。具体分为三类：正向反应生成，逆合成，溶剂预测。
* 正向反应：给定反应物的输入信息，预测产物。
* 逆合成：给定产物的输入信息，预测反应物。
* 溶剂：给定反应物的输入信息，预测相应的溶剂。

## 模型实现

### 硬件要求

- 支持 `Ascend` 后端，运行时可通过 `--device_target` 指定，默认 `Ascend`，可通过config配置。

### 版本依赖

- 需要安装 `MindSpore == 2.5.0`。

### 安装

- 安装 MindSpore：参考官方安装指南 `https://www.mindspore.cn/install`
- 安装依赖包：`pip install -r requirement.txt`

### 数据集与权重

模型训练好的权重以及数据集位于 [ReactGen_ckpt](https://ai.gitcode.com/AI4Science/ReactGen_ckpt)，

### 代码目录结构
```
ReactGen
|   infer_entry.py      # 模型推理统一入口
|   README.md           # README文件(英文)
|   README_CN.md        # README文件(中文)
|   requirements.txt    # 运行环境依赖
|   train_entry.py      # 模型训练统一入口
|   
+---config
|       forward_1.5b_inference.yaml         # 正向反应预测的推理配置
|       forward_1.5b_training.yaml          # 正向反应预测的训练配置
|       retrosynthesis_1.5b_inference.yaml  # 逆合成的推理配置
|       retrosynthesis_1.5b_training.yaml   # 逆合成的训练配置
|       solvent_1.5b_inference.yaml         # 溶剂预测的推理配置
|       solvent_1.5b_training.yaml          # 溶剂预测的训练配置
|       
+---dataset
|       dataset.py          # 正向反应、逆合成的数据处理
|       solvent_dataset.py  # 溶剂数据处理
|       
+---eval
|       evaluation_metrics.py       # 反应预测指标评估
|       
+---inference
|       inference_full_multi.py     # 正向反应、逆合成的推理脚本
|       inference_solvent_multi.py  # 溶剂预测的推理脚本
|       
+---model
|       reactionqwen.py             # ReactGen模型架构
|       reactionqwen_solvent.py     # ReactGen + classifier
|       tokenizer.py                # 分词器
|       
+---train
|       train_full_multi.py         # 正向反应、逆合成的训练脚本
|       train_solvent_multi.py      # 溶剂预测的训练脚本
|       
\---utils
        analyze_data_length.py      # 数据分布分析
        clean_solvent_dataset.py    # 溶剂数据集清洗
        create_solvent_map.py       # 处理稀有溶剂
        process_predictions.py      # 预测指标处理
        solvent_dataset_process.py  # 溶剂数据处理
        analyze_solvent.py          # 溶剂数据分析
        vocab_extension.py          # 词表拓展
```  

- 模型主体为 `MultiModalQwen`（`model/reactionqwen.py`），可通过 `train_entry.py`或`infer_entry.py`的程序统一入口进行模型的训练与推理。

## 模型运行步骤

### 训练

- 确保已完成以下准备：
    - 安装 MindSpore 及依赖包；
    - 下载权重与数据集；
    - 根据任务需求使用config下的 `xx_1.5b_training.yaml`，可根据需要进行配置：
- 在 `reactgen` 目录下执行：

```bash
python train_entry.py --config config/forward_1.5b_training.yaml
```

### 推理

- 将需要加载的权重路径写入config下的 `xx_1.5b_inference.yaml` 中 `inference.model_path` 字段。

- 在 `reactgen` 目录下执行：

```bash
python infer_entry.py --config config/forward_1.5b_inference.yaml
```

推理结果将以JSON形式保存在`results`目录下的`xx_metrics.json`文件中，格式为：

```
{
  "total_predictions": 39994,
  "valid_predictions": 39674,
  "exact_matches": 29363,
  "validity_rate": 0.991998799819973,
  "top1_accuracy": 0.7341851277691653,
  "valid_accuracy": 0.7401068709986389,
  "invalid_predictions": 320
}
```

## 许可证

- 开源协议：`Apache License 2.0`
- 许可证链接：`http://www.apache.org/licenses/LICENSE-2.0`