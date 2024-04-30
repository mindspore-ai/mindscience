# DeepH-E3

### Python

The python interpreter version needs to be at least 3.7. The following packages are also needed:

- Numpy
- mindspore 2.1 / mindspore 2.2
- pymatgen
- pathos
- h5py


## 快速运行
1. 从https://zenodo.org/records/7553640 下载  Bilayer_graphene_dataset.zip 到当前目录并解压， 不要修改其文件名。
2. 将mindchemistry文件包下载到该目录。
3. 修改configs/Bilayer_graphene_train.ini 下 save_dir ， processed_data_dir，  save_graph_dir 这三个路径为你自己的路径(即当前路径)
4. 运行网络: python deephe3_train.py configs/Bilayer_graphene_train.ini
5. 若要改用卡号，更改deephe3-train.py中的set_context()

### 注意
1. 该版本的deephe3和torch deephe3在模型架构上做了优化，在equiconv中，使用so2代替so3对卷积操作进行了优化，具体可以参考该论文 https://arxiv.org/abs/2302.03655
2. 目前暂时只支持batch size为1 ， 以及Bilayer_graphene_dataset， 其他数据集还需进行相应适配
3. 执行上面代码，实际执行了3步骤， 第一步数据预处理，只需要做一次，会自动保存到save_graph_dir，之后运行就会直接读二进制文件; 第二步图生成 ; 第三步模型训练
4. mindspore由于使用静态图加速， 在第一个epoch的时候会编译很久，通常会有几十分钟，后面开始就会很快。