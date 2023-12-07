# ESM-2

## 模型介绍

[ESM-2](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v1.full.pdf?utm_campaign=M2D2%20Community%20Round-Up&utm_medium=email&utm_source=Revue%20newsletter)是蛋白质语言模型。

ESM-2系列模型是迄今为止训练的最大的蛋白质语言模型，其参数仅比最近开发的最大文本模型少一个数量级。
ESM-2是一个基于transformer的语言模型，它使用注意力机制来学习输入序列中氨基酸对之间的相互作用。
ESM-2比以前的模型有了实质性的改进，即使在150M参数下，ESM-2也比在650M参数下的ESM-1生成语言模型捕获了更准确的结构图像。

相对于上一代模型ESM-1b，改进了模型体系结构、训练参数，并增加了计算资源和数据。添加相对位置嵌入可以推广到任意长度序列。这些修改导致了模型效果更好。
具有150M参数的ESM-2模型比具有650M参数的ESM-1b模型性能更好。在结构预测基准上，它的表现也优于其他最近的蛋白质语言模型。这种性能的提高与在大型语言建模领域建立的缩放定律一致。
15B参数ESM-2模型仅比已经训练过的最大最先进的文本语言模型小一个数量级，如Chinchilla（700亿参数）、GPT3和OPT-175B（都是1750亿参数）和PALM（5400亿参数）。

ESM-2的预训练模型采样的数据集为UR50/D 2021_04。

当前PipeLine中ESM-2只提供推理，暂不支持训练。

## 如何使用

EMS-2运行样例代码如下所示。

```bash
import numpy as np
from mindsponge.pipeline import PipeLine

pipeline = PipeLine('ESM2')
pipeline.initialize('config')
pipeline.model.from_pretrained()
data = [("protein3", "KA<mask>ISQ")]
kwargs = {"return_contacts": True}
_, _, _, contacts = pipeline.predict(data, **kwargs)
contacts = contacts.asnumpy()
tokens_len = pipeline.dataset.batch_lens[0]
attention_contacts = contacts[0]
matrix = attention_contacts[: tokens_len, : tokens_len]
print("contact map", matrix)
```

## 引用

```bash
@article{lin2022language,
  title={Language models of protein sequences at the scale of evolution enable accurate structure prediction},
  author={Lin, Zeming and Akin, Halil and Rao, Roshan and Hie, Brian and Zhu, Zhongkai and Lu, Wenting and Smetanin, Nikita and dos Santos Costa, Allan and Fazel-Zarandi, Maryam and Sercu, Tom and Candido, Sal and others},
  journal={bioRxiv},
  year={2022},
  publisher={Cold Spring Harbor Laboratory}
}
```
