# Evo2-7B-MindSpore

## 介绍

[Evo 2](https://github.com/arcinstitute/evo2) 是由 Arc Institute 联合 NVIDIA、斯坦福大学、加州大学伯克利分校与旧金山分校推出的 全球最大开源生物学 AI 模型，被誉为“生物界的 DeepSeek”。Evo2 首次把“生成式 AI”范式带到 序列层面，与 AlphaFold（结构层面）互补，构成“基因-蛋白-功能”全尺度计算基石，为精准医疗、合成生物学、基因治疗提供通用基础设施。

![Evo2coverpic](./docs/img/evo2.jpg)

本项目支持使用MindSpore进行Evo2-7B推理。

### 硬件要求

- Atlas 800T2 A2

### 软件要求

- Python >= 3.12
- CANN >= 8.2.rc1 （需要安装nnal包）
- MindSpore >= 2.8.0

## 前置准备
### 克隆仓库

```
git clone https://atomgit.com/mindspore-lab/mindscience.git
cd mindscience/MindSPONGE/research/Evo2
```

### 安装依赖
1. 新建conda环境变量，并安装以下依赖
```shell
conda install -c conda-forge binutils=2.38 --yes
pip install pytest ninja sympy matplotlib pyyaml tqdm einops rich torch biopython
```
2. 安装MindSpore >= 2.7.1
```shell
pip install mindspore==2.7.1 -i https://repo.mindspore.cn/pypi/simple --trusted-host repo.mindspore.cn --extra-index-url https://repo.huaweicloud.com/repository/pypi/simple --force-reinstall
```

3. MindSpore FFT依赖
- 需安装CANN>=8.2.rc1的nnal run包，并设置atb、asdsip环境变量：
```shell
# {PATH}为CANN安装路径
source {PATH}/nnal/atb/set_env.sh
source {PATH}/nnal/asdsip/set_env.sh
```
- 测试FFT算子可用性

```shell
# 克隆MindScience仓库
git clone https://gitee.com/mindspore/mindscience.git

# 测试FFT算子是否正常运行，用例应全部正常通过（显示PASS）
cd mindscience/tests/sciops
pytest test_asd_fft.py
```
附：昇腾FFT算子接入MindSpore特性issue：
https://gitee.com/mindspore/mindscience/issues/ICX22I

4. 映射环境变量
```bash
# {PATH}为mindscience/MindSPONGE/research/Evo2根目录
export PYTHONPATH={PATH}/evo2:{PATH}/vortex:{PATH}/mindscience
```

## 权重获取
本篇提供两种方式获取MindSpore可运行的Evo2-7B ckpt。

- 在线下载

通过魔方社区下载`evo2_7b_base_ms.ckpt`权重：https://modelers.cn/models/chen25/evo2-7b

注意：魔方社区仅上传evo2_7b_base，为7B参数量8K上下文的参数，若需要7B参数量1M上下文，需要通过下面权重转换方法。

- 权重转换

从[HuggingFace](https://huggingface.co/arcinstitute/evo2_7b_base)上获取`evo2_7b_base.pt`，放到`evo2/evo2/ckpt`文件夹，进行权重转换将pt文件转化为MindSpore支持的ckpt（可去掉一些层数加载以控制模型规模，详情可查看`convert_ckpt.py`文件）。
```bash
cd ckpt/
python convert_ckpt.py 
```
转换完后`ckpt/`下应有三个文件：
```shell
evo2_7b_base.pt evo2_7b_base_ms.ckpt convert_ckpt.py
```


## 运行推理

需要在`evo2/evo2/`文件夹下运行
```bash
cd evo2/evo2/
```

### Forward
Evo 2可对一段DNA序列上的每个位置，计算并给出该位置出现对应碱基的概率（似然度），从而完成对整条序列的概率评分。
```shell
python ../../examples/test_evo2_forward.py
```

### Embeddings
Evo 2的嵌入向量（embeddings）可以保存下来用于下游任务。论文中发现，中间层的嵌入向量比最后一层的嵌入向量效果更好，详见论文描述。
```shell
python ../../examples/test_evo2_embeddings.py
```

### Generation
Evo 2可以根据prompt生成DNA序列
```shell
python ../../examples/test_evo2_generation.py
```

### 许可证
详情请参阅[LICENSE](./evo2/LICENSE)文件

### 参考文献
- Brixi, G., Durrant, M.G., Ku, J. et al. Genome modelling and design across all domains of life with Evo 2. Nature (2026). https://doi.org/10.1038/s41586-026-10176-5

### 参考实现
- https://github.com/arcinstitute/evo2

### 贡献者
yanglong_unimelb, wenziyi2025, wuruifang2025, wujunchi2025, chenzhihui2025