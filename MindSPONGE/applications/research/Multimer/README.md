# AlphaFold-Multimer 蛋白质复合物结构预测工具

Multimer的主要功能是高精度预测蛋白质复合物结构，模型主要参考了AlphaFold-Multimer模型，推理长度支持与MEGA-Fold一致。

```bash
用法：python main.py --data_config ./config/data.yaml --model_config ./config/model.yaml --input_path INPUT_FILE_PATH_LIST --checkpoint_path CHECKPOINT_PATH

选项：
--data_config                   数据预处理参数配置
--model_config                  模型超参配置
--input_path                    输入multimer的序列文件(包括复合物的所有fasta文件，以list形式传入)
--checkpoint_path               AlphaFold-Multimer模型权重
```

## 参考文献

[1] Evans R, O’Neill M, Pritzel A, et al. Protein complex prediction with AlphaFold-Multimer[J]. BioRxiv, 2022: 2021.10. 04.463034.