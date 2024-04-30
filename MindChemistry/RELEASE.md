# MindSpore Chemistry Release Notes

[查看中文](./RELEASE_CN.md)

MindSpore Chemistry is a toolkit built on MindSpore endeavoring to enable the joint research of AI and chemistry with high efficiency and to seek th facilitate an innovative paradigm of joint research between AI and chemistry.

## MindSpore Chemistry 0.1.0 Release Notes

### Major Features

* Provide a **high-entropy alloy composition design approach**: Based on generation model and ranking model generating high-entropy alloy composition candidates and candidates' ranks, this approach constructs an active learning workflow for enabling chemistry experts to accelerate design of novel materials.
* Provide **molecular energy prediction models**: Based on equivariant computing library, the property prediction models NequIP and Allegro are trained effectively and infer molecular energy with high accuracy given atomic information.
* Provide an **electronic Structure Prediction model**: We integrate the DeephE3nn model, an equivariant neural network based on E3, to predict a Hamiltonian by using the structure of atoms.
* Provide a **crystalline material properties prediction model**: We integrate the Matformer model, based on graph neural networks and Transformer architectures, for predicting various properties of crystalline materials.
* Provide a **graph computing library**：We provide graph computing interface, such as graph dataloader and graph aggregation etc.
* Provide an **equivariant computing library**: We provide basic modules such as Irreps, Spherical Harmonics as well as user-friendly equivariant layers such as equivarant Activation and Linear layers for easy construction of equivariant neural networks.

### Contributors

Thanks goes to these wonderful people:

yufan, wangzidong, liuhongsheng, gongyue, gengchenhua, linghejing, yanchaojie, suyun, wujian, caowenbin

Contributions of any kind are welcome!
