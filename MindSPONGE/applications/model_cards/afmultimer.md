# AlphaFold Multimer

## 模型介绍

[AlphaFold Multimer](https://www.biorxiv.org/content/10.1101/2021.10.04.463034v2.abstract)是蛋白质复合物结构预测模型。

AlphaFold 2主要是通过添加连接子(Linker)将多链进行连接或残基间插入间隔等方法，将多链“伪装”成单链输入进行结构模拟。AlphaFold-Multimer在保留了AlphaFold 2算法一些重要特性的基础上，做了部分调整以满足复合物结合界面结构的特殊需要：

- 训练数据同来自于PDB，出于对计算和存储消耗的考虑，AlphaFold-Multimer同样沿用了AlphaFold 2将蛋白质截断为384个氨基酸长度输入方式，但截取的方法上力求扩大链覆盖度、截断片段多样性的同时，兼顾结合面与非接合面的截取。

- 沿用AlphaFold 2特色的FAPE(Frame aligned point error)结构打分函数，且为链内氨基酸对原子间设置截断距离为 10 埃，链间不设置固定截断距离值。

同时AlphaFold-Multimer也具备独特的计算方法，其中最具创新的模块在于多链特征提取和对称置换，使其超越包括基于AlphaFold的所有既有预测方法。

- 修改损失函数。同源多聚物将被视作同样的序列多次出现，充分考虑对称置换的情形，如预测一个A2B形式的复合物时，充分考虑亚单位所有的排列组合形式，包括两个A单位间置换的同等有效性等，避免正确的预测被罚分的情况，保证模型训练的有效性。

- 链间共进化。AlphaFold-Multimer采用了以往文献报道的方法，根据遗传距离或序列相似性判断种属关系，用同种属的序列进行配对以期获得同源结构及MSA信息并供给网络。

- 对位置编码(postional encoding)进行了重新编码，利用f_asym_id来进行链的编码，原来的残基距离编码d_ij只能在同一套f_asym_id中进行。利用f_entity_id进行实体编码（哪种蛋白质），利用f_sym_id来对同一套实体中的蛋白进行区分。

- 修改了模型置信度，提高了链间氨基酸残基间作用的权重，从而提升结合界面的精确度。

当前PipeLine中AlphaFold Multimer只提供推理，暂不支持训练。

## 如何使用

以6T36蛋白为例，Multimer运行样例代码如下所示。

```bash
import os
import stat
import pickle
from mindsponge import PipeLine
from mindsponge.common.protein import to_pdb_v2, from_prediction_v2

cmd = "wget https://download.mindspore.cn/mindscience/mindsponge/Multimer/examples/6T36.pkl"
os.system(cmd)

pipe = PipeLine(name="Multimer")
pipe.set_device_id(0)
pipe.initialize("predict_256")
pipe.model.from_pretrained()
with open("./6T36.pkl", "rb") as f:
    raw_feature = pickle.load(f)
final_atom_positions, final_atom_mask, confidence, b_factors = pipe.predict(raw_feature)
unrelaxed_protein = from_prediction_v2(final_atom_positions,
                                       final_atom_mask,
                                       raw_feature["aatype"],
                                       raw_feature["residue_index"],
                                       b_factors,
                                       raw_feature["asym_id"],
                                       False)
pdb_file = to_pdb_v2(unrelaxed_protein)
os.makedirs('./result/', exist_ok=True)
os_flags = os.O_RDWR | os.O_CREAT
os_modes = stat.S_IRWXU
pdb_path = './result/unrelaxed_6T36.pdb'
with os.fdopen(os.open(pdb_path, os_flags, os_modes), 'w') as fout:
    fout.write(pdb_file)
print("confidence:", confidence)
```

## 引用

```bash
@article {AlphaFold-Multimer2021,
  author       = {Evans, Richard and O{\textquoteright}Neill, Michael and Pritzel, Alexander and Antropova, Natasha and Senior, Andrew and Green, Tim and {\v{Z}}{\'\i}dek, Augustin and Bates, Russ and Blackwell, Sam and Yim, Jason and Ronneberger, Olaf and Bodenstein, Sebastian and Zielinski, Michal and Bridgland, Alex and Potapenko, Anna and Cowie, Andrew and Tunyasuvunakool, Kathryn and Jain, Rishub and Clancy, Ellen and Kohli, Pushmeet and Jumper, John and Hassabis, Demis},
  journal      = {bioRxiv},
  title        = {Protein complex prediction with AlphaFold-Multimer},
  year         = {2021},
  elocation-id = {2021.10.04.463034},
  doi          = {10.1101/2021.10.04.463034},
  URL          = {https://www.biorxiv.org/content/early/2021/10/04/2021.10.04.463034},
  eprint       = {https://www.biorxiv.org/content/early/2021/10/04/2021.10.04.463034.full.pdf},
}
```
