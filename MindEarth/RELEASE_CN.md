# MindEarth 0.1.0 Release Notes

Initial release of MindEarth.

## 主要特性

### 中期天气预报预测

* FourCastNet

FourCastNet，傅立叶神经网络预测的缩写，是一个全球数据驱动的天气预测模型，以0.25∘分辨率提供准确的中短期全球预测。FourCastNet准确预测高分辨率、快速的时间尺度变量，如表面风速、降水量和大气水蒸气。它对规划风能资源、预测热带气旋、温带气旋和大气河流等极端天气事件具有重要意义。

* ViT-KNO

KNO是一个在准确性和效率方面优于其他现有技术且具有代表性的证明。除了原始版本的KNO之外，还有多种基于不同神经网络架构的KNO新变体，以提高我们模块的通用性，例如ViT-KNO。这些变体通过在具有代表性的偏微分方程上实施的网格无关和长期预测实验进行了验证。

* GraphCast

GraphCast是一种新的基于ML的天气仿真器，它超越了世界上最精确的确定性中期天气预报系统和所有ML基线。GraphCast自回归模型是使用来自欧洲中期天气预报中心（ECMWF）的ERA5再分析气象数据训练的，该模型基于神经网络和一种新的高分辨率多尺度网格表示。在赤道它的分辨率约为25×25km，每10小时可以为5个表面变量和37个垂直压力面的6个大气变量创建一个6天的预报。

### 贡献者

Thanks goes to these wonderful people:

yufan, wangzidong, liuhongsheng, zhouhongye, liulei, libokai, chengqiang, dongyonghan, zhouchuansai

Contributions of any kind are welcome!
