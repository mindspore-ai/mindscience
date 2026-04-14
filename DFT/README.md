# 密度泛函套件
dataset下载地址：
DeepDFT：支持下面三个数据集，建议选择NMC数据集，仅为10GB左右。下载后将数据文件放置在`data/nmc/`目录下。
* QM9下载地址：https://data.dtu.dk/articles/dataset/QM9_Charge_Densities_and_Energies_Calculated_with_VASP/16794500
* NMC下载地址：https://data.dtu.dk/articles/dataset/NMC_Li-ion_Battery_Cathode_Energies_and_Charge_Densities/16837721
* Eth下载地址：https://data.dtu.dk/articles/dataset/Ethylene_Carbonate_Molecular_Dynamics/16691825
ML-DFT：https://github.com/Ramprasad-Group/ML-DFT/tree/main/tutorials/database，下载后放置到`dataset`目录下
Delta-DFT：https://github.com/MihailBogojeski/ml-dft/tree/master/water_102，下载后放置于`Dataset`目录下。
DeepH：从 https://zenodo.org/records/7553640 下载 Bilayer_graphene_dataset.zip 到`deephe3nn/`目录下并解压，不要修改其文件名。
E3-DNA：从 http://aisccc.cn/database/data-details?id=171&type=resource 下载 dataset.zip 并解压，将`.pkl`文件移动到`data`目录下。
ML-DFTXC：从 https://github.com/zhouyyc6782/oep-wy-xcnn/tree/master/example/simple_H2 下载 `H2_0.9_9.npy` 和 `H2_d0700.str` 到 `data` 目录下。

ckpt：
DeepDFT ckpt下载地址：https://github.com/12138xs/MindDFT/tree/main/DeepDFT/checkpoints
下载后将各目录ckpt放置在`checkpoints`目录下