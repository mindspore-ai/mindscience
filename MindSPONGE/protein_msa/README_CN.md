# Protein MSA数据库
## Protein MSA数据库简介
 针对蛋白质的[多序列比对](https://en.wikipedia.org/wiki/Multiple_sequence_alignment)（multiple sequence alignment; MSA）是研究蛋白质的结构、功能和进化关系等问题的重要方法。MSA数据中蕴含了构成蛋白质的氨基酸序列中的保守性质(conservation)、协同突变(co-evolution)和功能与物种进化关系(phylogenetics)的相关信息。<br>
<!--
![MSA与蛋白质性质的关联](https://gitee.com/jz_90/mindscience/tree/master/MindSPONGE/docs/MSA_Figure.png)<br>
-->
 人类已知的存在于自然界中的蛋白质序列数目已经上亿并在快速增长，但仅凭这些蛋白质单序列的数据很难了解蛋白之间的关系。Protein MSA数据库，就是一个对不同蛋白质序列之间的关系进行了标记的大规模“关系型”数据库，被标记为关联的蛋白质序列之间的相似度、进化关系和突变所在位点的分布等信息对蛋白质结构和功能的预测极为重要。例如在AlphaFold2模型[1]中，目标蛋白序列的MSA信息就是预测结构的必要输入信息之一。
## 数据库建立方法
 Protein MSA中的目标序列将几乎完全覆盖最新版本（2021.02发布）的[UniRef50数据库](https://www.uniprot.org/uniref/)中的蛋白质序列，而比对序列来自于最新版本的[UniClust30数据库](http://wwwuser.gwdg.de/~compbiol/uniclust/2020_06/)。对于每条来自UniRef50数据库中的目标序列，我们采用HHBlits算法在UniClust30数据库进行搜索和比对，并将检索以文本形式存放于Protein MSA数据仓库下Raw_Data目录下。
## 数据库规模
 Protein MSA数据库中包含的目标序列约有50M条，之后还将继续扩展和更新。对于每条目标序列，其比对序列的平均条数（或MSA深度）大于1000，因此该数据集里以MSA的形式汇总了超过50B条蛋白质序列（包括了一些重复出现的比对序列）。
## 使用场景
 从科学应用的角度看，MSA的数量和质量很大程度上影响了目前最先进的结构模型的预测速度和精度，而且产生MSA的非参数化算法仍是诸多蛋白预测方法中主要决速步之一。因此，Protein MSA数据库本身可以作为这些结构模型的预训练材料，用来挖掘序列信息甚至快速生成新的序列特征，这对解决研究、设计蛋白质中所面临的高变异序列和孤儿序列等问题具有巨大的潜在价值。为了便于AI领域的研究人员直接使用，Protein MSA原始数据还会被转化为浮点数类型压缩存储，并对已有的AI框架如MindSpore上提供数据接口的支持。我们鼓励并期待来自生物信息学、数据科学和自然语言处理等AI研究领域的专家和人才充分碰撞与合作，引入、改进或设计全新的AI模型，来充分地挖掘Protein MSA数据集中所隐藏的“大自然的秘密”。
## 使用与下载方法
 ToDo: 
## 许可与引用
 ToDo: 
## 维护、更新与社区贡献方式
 ToDo: 
