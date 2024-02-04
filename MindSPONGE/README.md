 ENGLISH | [简体中文](README_CN.md)

![MindSPONGE LOGO](docs/MindSPONGE.png "MindSPONGE logo")

[![LICENSE](https://img.shields.io/github/license/mindspore-ai/mindspore.svg?style=flat-square)](https://github.com/mindspore-ai/mindspore/blob/master/LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://gitee.com/mindspore/mindscience/pulls)
[![docs 1.0.0-alpha](https://img.shields.io/badge/docs-1.0.0--alpha-blueviolet.svg?style=flat-square)](https://mindspore.cn/mindsponge/docs/zh-CN/r1.0.0-alpha/index.html)
[![release](https://img.shields.io/badge/release-1.0.0--alpha-blueviolet.svg?style=flat-square)](https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/RELEASE_CN.md)

# **MindSpore SPONGE**

## **Introduction**

MindSpore SPONGE(Simulation Package tOwards Next GEneration molecular modelling) is a toolkit for Computational Biology based on AI framework [MindSpore](https://www.mindspore.cn/)，which supports MD, folding and so on. It aims to provide efficient AI computational biology software for a wide range of scientific researchers, staff, teachers and students.

<div align=center><img src="docs/archi.png" alt="MindSPONGE Architecture" width="700"/></div>

## **Latest News** 📰

- 🔥`2023.12.07` The antibody design model Tiangong won the "2023 AIIA Top Ten Pioneer Application Cases of Artificial Intelligence", [Related News](https://mp.weixin.qq.com/s/UQStKzm0fdXbA4RQgLE8fw)
- 🔥`2023.11.10` MEGA-EvoGen Paper "Unsupervisedly Prompting AlphaFold2 for Accurate Few-Shot Protein Structure Prediction" is published in JCTC. Please refer to [paper](https://pubs.acs.org/doi/10.1021/acs.jctc.3c00528?cookieSet=1) and [code](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/applications/MEGAProtein).
- 🔥`2023.8.21—2023.8.25` MindSpore SPONGE SIG [**Summer School**](https://mp.weixin.qq.com/s/oOaJ9KlUnWbptZWqSvam7g) is coming soon !
- 🔥 [**open source internship task**](https://gitee.com/mindspore/community/issues/I561LI?from=project-issue) has been released! Everyone is welcome to claim it~
- 🔥`2023.6.26` MindSPONGE Paper "Artificial Intelligence Enhanced Molecular Simulations" is published in JCTC and achieve Most Read Articles. Please refer to [paper](https://pubs.acs.org/doi/10.1021/acs.jctc.3c00214).
- 🔥`2023.5.31` Paper "Assisting and Accelerating NMR Assignment with Restrained Structure Prediction" is preprinted in arxiv, Please refer to [paper](https://arxiv.org/abs/2208.09652) and [code](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/applications/research/FAAST/).
- `2023.1.31` MindSPONGE version 1.0.0-alpha is released. The documents are available on [**Scientific Computing MindSPONGE module**](https://mindspore.cn/mindsponge/docs/en/r1.0.0-alpha/index.html) on MindSpore website
- `2022.8.23` Paper "Few-Shot Learning of Accurate Folding Landscape for Protein Structure Prediction" is preprinted in arxiv, Please refer to [paper](https://arxiv.org/abs/2208.09652)
- `2022.8.11—2022.8.15` MindSpore SPONGE SIG [**Summer School**](#special-interesting-group-), [**replay**](https://www.bilibili.com/video/BV1pB4y167yS?spm_id_from=333.999.0.0&vd_source=94e532d8ff646603295d235e65ef1453)
- `2022.07.18` Paper "SPONGE: A GPU-Accelerated Molecular Dynamics Package with Enhanced Sampling and AI-Driven Algorithms"is published in Chinese Journal of Chemistry. Please refer to [paper](https://onlinelibrary.wiley.com/doi/epdf/10.1002/cjoc.202100456) and [codes](https://gitee.com/mindspore/mindscience/tree/dev-md/MindSPONGE/applications/molecular_dynamics)
- `2022.07.09` MEGA-Assessment wins CAMEO-QE monthly 1st
- `2022.06.27` Paper "PSP: Million-level Protein Sequence Dataset for Protein Structure Prediction" is preprinted in arxiv. Please refer to [paper](https://arxiv.org/pdf/2206.12240v1.pdf) and [codes](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/applications/MEGAProtein).
- `2022.04.21` MEGA-Fold wins CAMEO-3D monthly 1st. [Related News](https://www.huawei.com/cn/news/2022/4/mindspore-cameo-protein-ascend)

## **Quick Start**

### Protein Multimer Structure Prediction

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
f = open("./6T36.pkl", "rb")
raw_feature = pickle.load(f)
f.close()
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

<div align=left>
    <img src="docs/multimer.gif" width=30%>
</div>

**More Cases**：👀

- [NMR Data Automatic Analysis FAAST](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/applications/research/FAAST)
- [Protein Structure Prediction MEGA-Fold](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/applications/MEGAProtein/)
- [Protein Structure Assessment MEGA-Assessment](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/applications/MEGAProtein/)
- [Evolution Engine MEGA-EvoGen](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/applications/MEGAProtein/)
- Function based protein design (TO BE DONE)
- Structure based protein design (TO BE DONE)
- Protein function prediction (TO BE DONE)
- Molecular representation model (TO BE DONE)

## **Installation**

### Hardware

| Hardware      | OS              | Status |
| :------------ | :-------------- | :--- |
| Ascend 910    | Ubuntu-x86      | ✔️ |
|               | Ubuntu-aarch64  | ✔️ |
|               | EulerOS-aarch64 | ✔️ |
|               | CentOS-x86      | ✔️ |
|               | CentOS-aarch64  | ✔️ |
| GPU CUDA 10.1 | Ubuntu-x86      | ✔️ |

- CUDA>=10.1
- Ubuntu>=16.04

### **pip install**

- Ascend backend

```bash
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.2.1/MindScience/mindsponge/ascend/aarch64/mindsponge_ascend-1.0.0rc2-py3-none-any.whl
```

- GPU backend

```bash
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.2.1/MindScience/mindsponge/gpu/x86_64/cuda-10.1/mindsponge_gpu-1.0.0rc2-py3-none-any.whl
```

The version of mindsponge installed by pip corresponds to the r0.5 branch code. The code can be downloaded using the following instruct.

```bash
git clone -b r0.5 https://gitee.com/mindspore/mindscience.git
```

### **source code install**

```bash
git clone https://gitee.com/mindspore/mindscience.git
cd {PATH}/mindscience/MindSPONGE
```

- Ascend backend

```bash
bash build.sh -e ascend
```

- GPU backend

```bash
export CUDA_PATH={your_cuda_path}
bash build.sh -e gpu -j32
```

- Install whl package

```bash
cd {PATH}/mindscience/MindSPONGE/output
pip install mindsponge_ascend*.whl # Ascend
pip install mindsponge-gpu*.whl # GPU
```

### API

For details about MindSPONGE APIs, please refer to [API](https://mindspore.cn/mindsponge/docs/en/master/index.html) pages.

## **Community**

### CO-CHAIR

<div align=center>
    <a href="https://gitee.com/helloyesterday">
        <img src="docs/co-chair/yangyi.jpg" width=15%>
    </a>
    &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;
    <a href="https://gitee.com/jz_90">
        <img src="docs/co-chair/zhangjun.jpg" width=15%>
    </a>
    &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;
    <a href="https://gitee.com/sirui63">
        <img src="docs/co-chair/sirui.jpg" width=15%>
    </a>
    <br/>
    <font>Shenzhen Bay Laboratory Yi Isaac Yang</font>
    &emsp;&emsp;&emsp;
    <font>Chang Ping Laboratory Jun Zhang</font>
    &emsp;&emsp;&emsp;
    <font>Chang Ping Laboratory Sirui Liu</font>
</div>

### Special Interesting Group 🏠

MindSpore SPONGE SIG (Special Interesting Group) is a team composed of a group of people who are interested and have a mission to make achievements in the field of AI × biological computing.

MindSpore SPONGE SIG group provides efficient and easy-to-use AI computational biology software for researchers, teachers and students, and provides a platform for people with strong abilities or strong interests in this field to communicate and cooperate together.

At present, the SIG group has six core teachers. After members joining the SIG group, our teachers will lead the team to carry out scientific research and develop the software function development. Of course, members are also welcome to do research on their own topics using MindSPONGE.

In the SIG group, we will hold various activities, including summer school, public lecture, technology communication meeting and other large-scale activities. Small-scale activities like weekly meetings, blogs writing will also be held in the group. By joining the activities, there will be lots of chances to communicate with our experts. During the summer school program ended on August 15th, we invited 13 teachers to have a five-day lecture mainly including three themes of MindSpore basics, molecular dynamics and advanced AI × Science courses. You can get the replay [here](https://www.bilibili.com/video/BV1pB4y167yS?spm_id_from=333.999.0.0&vd_source=94e532d8ff646603295d235e65ef1453).

In the SIG group, we will also release the public intelligence task and [open source internship task](https://gitee.com/mindspore/community/issues/I561LI?from=project-issue), welcome everyone to claim it.

If you want to join us and become a member of our group, please send your resume to liushuo65@huawei.com, we are always looking forward to your arrival.

### Core Contributor 🧑‍🤝‍🧑

- [Yi Qin Gao Research Group](https://www.chem.pku.edu.cn/gaoyq/):  [Yi Isaac Yang](https://gitee.com/helloyesterday)，[Jun Zhang](https://gitee.com/jz_90)，[Sirui Liu](https://gitee.com/sirui63)，[Yijie Xia](https://gitee.com/xiayijie)，[Diqing Chen](https://gitee.com/dechin)，[Yu-Peng Huang](https://gitee.com/gao_hyp_xyj_admin)

### Cooperative Partner

<div class="item1">
    <img src="docs/cooperative_partner/北京大学.png" width=20%>
    &emsp;
    <img src="docs/cooperative_partner/深圳湾.jpg" width=20%>
    &emsp;
    <img src="docs/cooperative_partner/西电.png" width=20%>
</div>

## **Contribution Guide**

- Please click here to see how to contribute your code:[Contribution Guide](https://gitee.com/mindspore/mindscience/blob/master/CONTRIBUTION.md)

## **License**

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)
