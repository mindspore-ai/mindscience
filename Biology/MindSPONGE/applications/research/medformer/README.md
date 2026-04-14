# MedFormer: Transformer-based Drug Perturbation Prediction

MedFormer is a drug perturbation prediction framework based on the Transformer architecture, designed to predict the transcriptional responses of small molecule drugs under different cellular states. By integrating drug molecular fingerprints, baseline transcriptional states, and gene embeddings, it achieves high-precision predictions for unseen drugs and cell types, and is scalable to single-cell data.  

This project is based on [MindSPONGE](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE) and implemented in Python.

---

## 🔧 requirement

- Python 3.8+

- mindspore >= 3.9.0

- numpy

- pandas

- scikit-learn

- rdkit

- tqdm  

---

## Quick start

Raw data link:
https://zenodo.org/records/14230870

Essential data link:
https://pan.baidu.com/s/1AKJT6gvSf05PgYit6SPbYQ?pwd=f5iy

Run：
`python train.py --split_key drug_split_0 --ablation False --device_id 0`

`split_key` indicates which fold of the k-fold cross-validation should be used as the training set.

`ablation` indicates whether an ablation experiment is to be conducted.

`device_id` represents the ID of the computing card being used. It can be filled in according to the actual situation. By default, the idle computing card among all available ones will be selected automatically.