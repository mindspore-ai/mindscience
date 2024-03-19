# MindSpore Earth Release Notes

[ENGLISH](RELEASE.md) | 简体中文

MindSpore Earth是基于昇思MindSpore开发的地球科学套件，支持多时空尺度气象预报、数据前后处理等任务，致力于高效使能AI+气象海洋的融合研究。

## MindSpore Earth 0.2.0 Release Notes

### 主要特性和增强

#### 短临降水预测

- [STABLE] [NowcastNet](https://gitee.com/mindspore/mindscience/tree/master/MindEarth/applications/nowcasting/Nowcastnet): 新增基于物理约束的生成式短临降水模型，支持未来3小时的雷达外推降水预报。

#### 中期天气预报预测

- [STABLE] [GraphCast](https://gitee.com/mindspore/mindscience/tree/master/MindEarth/applications/medium-range/graphcast): 新增支持0.25°高分辨率数据，GraphCast全尺度模型训练与推理。
- [STABLE] [FuXi](https://gitee.com/mindspore/mindscience/tree/master/MindEarth/applications/medium-range/fuxi): 新增气象模型伏羲，支持中期预报模型的训练和推理。
- [STABLE] [Medium Precipitation](https://gitee.com/mindspore/mindscience/tree/master/MindEarth/applications/medium-range/graphcast): 新增中期降水模块案例，支持中期的降水量预报，需要基于预训练的中期大模型做为backbone。
- [RESEARCH] [CTEFNet](https://gitee.com/mindspore/mindscience/tree/master/MindEarth/applications/climate-prediction/ensoforecast): 新增基于 CNN 与迁移学习的厄尔尼诺预测模型，将ENSO有效预报时长延长至19个月。

### 贡献者

感谢以下开发者做出的贡献:

hsliu_ustc, hong-ye-zhou, liulei277, kevinli123, Zhou Chuansai, alancheng511, Cui Yinghao, xingzhongfan, cmy_melody

欢迎任何形式的贡献！

## MindSpore Earth 0.1.0 Release Notes

### 主要特性和增强

#### 短临降水预测

- [STABLE] [Dgmr](https://gitee.com/mindspore/mindscience/tree/master/MindEarth/applications/nowcasting/dgmr): Dgmr（雷达网络深度生成模型）是由DeepMind的研究人员开发的雷达降水概率临近预报的深度生成模型。该模型的主体是一个生成器，配合时间和空间判别器损失以及额外的正则化项进行对抗训练。模型从前四帧雷达序列学习上下文表示，用作采样器的输入，采样器是一个由卷积门控循环单元（GRU）构成的递归网络，它将上下文表示和从高斯分布中取样的潜向量作输入，对未来18个雷达场进行预测。

#### 中期天气预报预测

- [STABLE] [FourCastNet](https://gitee.com/mindspore/mindscience/tree/master/MindEarth/applications/medium-range/fourcastnet): FourCastNet采用自适应傅里叶神经算子AFNO，这种神经网络架构是对Vision Transformer模型的改进，它将混合操作步骤构建成连续的全局卷积，在傅里叶域中通过FFT有效实现，将空间混合复杂度降低到O(NlogN)。该模型为第一个预报精度能与欧洲中期天气预报中心（ECMWF）的高分辨率综合预测系统（IFS）模型比较的AI预报模型。
- [STABLE] [ViT-KNO](https://gitee.com/mindspore/mindscience/tree/master/MindEarth/applications/medium-range/koopman_vit): Koopman Neural Operator是一个基于Koopman的全局线性化理论并结合神经算子思想设计的一个轻量化的、网格无关的模型。该模型由华为先进计算与存储实验室与清华大学合作推出。通过在线性结构中嵌入复杂的动力学来约束重建过程，此模型能够捕获复杂的非线性行为，同时保持模型轻量级和计算有效性。与FNO相比，KNO具有更高效的训练性能与更优的预测精度。
- [STABLE] [GraphCast](https://gitee.com/mindspore/mindscience/tree/master/MindEarth/applications/medium-range/graphcast): GraphCast由谷歌DeepMind提出，该模型使用GNN在“编码-处理-解码”架构中自回归地生成预报结果。编码器将历史时刻的气象要素的纬度-经度输入网格映射到多尺度正二十面体网格表示；处理器在多网格表示上执行多轮消息传递；解码器将多网格表示映射回纬度-经度网络，作为下一时间步骤的预测。另外，针对多部预测精度衰减，MindEarth实现了多步迭代训练，以降低模型误差累积。

#### 数字高程模型超分

- [STABLE] [DEM-SRNet](https://gitee.com/mindspore/mindscience/tree/master/MindEarth/applications/dem-super-resolution): DEM-SRNet是一个数字高程模型的超分辨率模型，该模型基于30m分辨率的NASADEM卫星影像、联合国政府间海洋学委员会的450m分辨率GEBCO_2021公开数据和部分区域高分辨率海洋地形数据，采用深度残差预训练神经网络和迁移学习（Transfer Learning）相结合技术，生成全球90m高分辨率DEM。该数据集可以提供更加准确的基础地理信息，在全球气候变化、海洋潮汐运动、地球圈物质交换等研究领域发挥着至关重要的作用。

### 贡献者

感谢以下开发者做出的贡献:

hsliu_ustc, hong-ye-zhou, liulei277, kevinli123, alancheng511

欢迎任何形式的贡献！