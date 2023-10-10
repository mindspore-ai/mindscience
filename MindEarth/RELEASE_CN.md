# MindEarth 0.1.0 Release Notes

Initial release of MindEarth.

## 主要特性

### 数字高程模型超分

* DemNet

DEMNet是数字高程模型网络的简称，可以提供准确的基础地理数据，因此在全球气候变化、海洋潮汐运动、地球圈物质交换等研究领域发挥着至关重要的作用。全球海洋DEM是海洋地质学和海洋测绘的前沿分支，为了解海底构造运动和海底形成过程提供直接参考。

### 中期天气预报预测

* FourCastNet

FourCastNet采用自适应傅里叶神经算子AFNO，这种神经网络架构是对Vision Transformer模型的改进，他将混合操作步骤构建成连续的全局卷积，在傅里叶域中通过FFT有效实现，将空间混合复杂度降低到O(NlogN)。该模型为第一个预报精度能与欧洲中期天气预报中心（ECMWF）的高分辨率综合预测系统（IFS）模型比较的AI预报模型。

* ViT-KNO

Koopman Neural Operator是一个基于Koopman的全局线性化理论并结合神经算子思想设计的一个轻量化的、网格无关的模型。该模型由华为先进计算与存储实验室与清华大学合作推出。通过在线性结构中嵌入复杂的动力学来约束重建过程，此模型能够捕获复杂的非线性行为，同时保持模型轻量级和计算有效性。与FNO相比，KNO具有更高效的训练性能与更优的预测精度。

* GraphCast

GraphCast由谷歌DeepMind提出，该模型使用GNN在“编码-处理-解码”架构中自回归地生成预报结果。编码器将历史时刻的气象要素的纬度-经度输入网格映射到多尺度正二十面体网格表示；处理器在多网格表示上执行多轮消息传递；解码器将多网格表示映射回纬度-经度网络，作为下一时间步骤的预测。另外，针对多部预测精度衰减，MindEarth实现了多步迭代训练，以降低模型误差累积。

### 短临降水预报

* Dgmr

DgmrNet（雷达网络深度生成模型）是由DeepMind的研究人员开发的雷达降水概率临近预报的深度生成模型。它可以对面积达1,536公里×1,280公里的区域进行现实且时空一致的预测，并且提前时间为5至90分钟。该方法由50多名专业气象学家进行系统评估，结果表明，与两种竞争方法相比，DgmrNet在89%的情况下以其准确性和实用性排名第一。它可以提供概率预测，提高预测价值并支持操作实用性，并且在分辨率方面优于替代方法。

### 贡献者

Thanks goes to these wonderful people:

yufan, wangzidong, liuhongsheng, zhouhongye, liulei, libokai, chengqiang, dongyonghan, zhouchuansai

Contributions of any kind are welcome!
