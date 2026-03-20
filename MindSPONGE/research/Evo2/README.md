# [WIP] MindSpore-Evo2

## 介绍

[Evo2](https://github.com/arcinstitute/evo2) 是由 Arc Institute 联合 NVIDIA、斯坦福大学、加州大学伯克利分校与旧金山分校推出的 全球最大开源生物学 AI 模型，被誉为“生物界的 DeepSeek”。Evo2 首次把“生成式 AI”范式带到 序列层面，与 AlphaFold（结构层面）互补，构成“基因-蛋白-功能”全尺度计算基石，为精准医疗、合成生物学、基因治疗提供通用基础设施。

本项目支持使用MindSpore训推Evo2。

### RoadMap

- 支持Evo2 7b推理 [WIP]
- 支持Evo2训练 [PENDING]

## 环境 

### Ascend + MindSpore

1. 基础依赖
- Python >= 3.12
- CANN >= 8.2.rc1 （注意需要安装nnal包）
- MindSpore >= 2.7.1

2. Pip依赖
```bash
> pip install -r requirements.txt
```

3. MindSpore FFT依赖
- 克隆MindScience仓库
```bash
> git clone https://gitee.com/mindspore/mindscience.git
```
- 昇腾FFT算子接入MindSpore特性issue：
https://gitee.com/mindspore/mindscience/issues/ICX22I
- 测试FFT算子是否正常运行，用例应全部正常通过：
```
> cd mindscience/tests/sciops
> pytest test_asd_fft.py
```


4. 映射环境变量
```bash
> export PYTHONPATH=$PWD/evo2:$PWD/vortex:$PWD/mindscience
```

### GPU + Torch [[ref]](https://github.com/arcinstitute/evo2) （torch-gpu分支）

1. Python 3.12+
2. CUDA: 12.1+
3. cuDNN: 9.3+
4. GCC 9+

注意：transformer_engine的fp8相关功能不在当前项目考虑范围内

## 权重获取

从[HuggingFace](https://huggingface.co/arcinstitute/evo2_7b_base)上获取evo2_7b_base.pt，放到evo2/evo2/ckpt文件夹：
```bash
> ls evo2/evo2/ckpt
evo2_7b_base.pt
```

### MindSpore + Ascend
进行权重转换将pt文件转化为MindSpore支持的ckpt（可在convert_ckpt.py中去掉一些层数加载）。
```bash
> python convert_ckpt.py 
> ls
evo2_7b_base.pt evo2_7b_base_ms.ckpt
```

### Torch + GPU
无需其他操作

## 运行推理

需要在evo2/evo2/文件夹下运行
```bash
> cd evo2/evo2/
```

### 1. forward
```bash
> python ../../examples/test_evo2_forward.py
```
成功返回
```
Logits:  tensor([[[ -8.0625, -28.1250, -28.1250,  ..., -28.1250, -28.1250, -28.1250],
         [ -5.2812, -27.2500, -27.2500,  ..., -27.2500, -27.2500, -27.2500],
         [ -5.3438, -27.2500, -27.2500,  ..., -27.2500, -27.2500, -27.2500],
         [ -5.6250, -27.8750, -27.8750,  ..., -27.8750, -27.8750, -27.8750]]],
       device='cuda:0', dtype=torch.bfloat16)
Shape (batch, length, vocab):  torch.Size([1, 4, 512])
```

### 2. embeddings
```bash
> python ../../examples/test_evo2_embeddings.py
```
成功返回
```
Embeddings shape:  torch.Size([1, 4, 4096])
```

### 3. generation
```bash
> python ../../examples/test_evo2_generation.py
```
成功返回
```
Initializing inference params with max_seqlen=404
/ms_test2/lyy/Evo2/vortex/vortex/model/engine.py:559: UserWarning: Casting complex values to real discards the imaginary part (Triggered internally at /pytorch/aten/src/ATen/native/Copy.cpp:308.)
  inference_params.state_dict[layer_idx] = state[..., L - 1].to(dtype=state_dtype)
Prompt: "ACGT",	Output: "TATGTAATTTGCAAGCATTTATCGAAGCGTTTATCAATCAGAAAGGTGAAGCTTTAAAACTCCTCCAATGGCCTATCGGAAATTTCAGATATTGTCATACAAATTCCAGCATTCACATTACGCAACAAGCAAGAGAATCACGATACAGCAAGACTGTATATTGGAAGCCAGAGGTTAAAATTAACAATCATAAGTCAATGCTTAAAAATCACAGGGTCAATGCGGTCAAAAGTGCTCGTAATAAAAGACGAAGGTTTCAATCGATGAATTCACTTGCGTGGATGGGAGGAGCGCGCTCGTGACGTGTGTAGCCTATAGTGTGACAAAAGCCAAATAAAAGACATTCATGACAGTTAACAAGCAGCCCATAGCGAAGATTCTCGTGGTAGGTACTGTACCA",	Score: -1.3534809350967407
TATGTAATTTGCAAGCATTTATCGAAGCGTTTATCAATCAGAAAGGTGAAGCTTTAAAACTCCTCCAATGGCCTATCGGAAATTTCAGATATTGTCATACAAATTCCAGCATTCACATTACGCAACAAGCAAGAGAATCACGATACAGCAAGACTGTATATTGGAAGCCAGAGGTTAAAATTAACAATCATAAGTCAATGCTTAAAAATCACAGGGTCAATGCGGTCAAAAGTGCTCGTAATAAAAGACGAAGGTTTCAATCGATGAATTCACTTGCGTGGATGGGAGGAGCGCGCTCGTGACGTGTGTAGCCTATAGTGTGACAAAAGCCAAATAAAAGACATTCATGACAGTTAACAAGCAGCCCATAGCGAAGATTCTCGTGGTAGGTACTGTACCA
```

## 参与贡献

1. Fork 本仓库
2. 提交代码
3. 新建 Pull Request

### 贡献者

longyangyang, wujunchi2025, wenziyi2025, chenzhihui2025, wuruifang2025