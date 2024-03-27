# 数据生成

安装 Python 依赖：

```bash
pip3 install h5py matplotlib
```

数据生成使用 Dedalus 包，按照 [这里](https://dedalus-project.readthedocs.io/en/latest/pages/installation.html) 的说明安装。
安装完成后，可运行脚本文件生成数据。例如，使用如下命令生成系数范围 $[-3,3]$、随机种子为 1 的预训练数据集：

```bash
OMP_NUM_THREADS=1 python3 custom_sinus.py -m 3 -r 1
```

可使用如下命令查看命令行参数列表：

```bash
python3 custom_sinus.py -h
```

## 文件目录结构

```text
./
│  common.py                                 # 通用组件
│  custom_multi_component.py                 # 多分量方程数据生成主文件
│  custom_sinus.py                           # 带三角函数项方程数据生成主文件
│  custom_wave.py                            # 波方程数据生成主文件
│  inverse_sinus.py                          # 带三角函数项方程反问题数据生成主文件
│  inverse_wave.py                           # 波方程反问题数据生成主文件
│  README.md                                 # 数据集生成说明文档（英文）
│  README_CN.md                              # 数据集生成说明文档（中文）
```
