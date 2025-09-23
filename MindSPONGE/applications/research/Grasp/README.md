# Multimer/Grasp多卡并行推理

## 1 环境依赖

### 硬件依赖
- Atlas 800T A2 (64G)

### 软件依赖
- Python>=3.9
- CANN>=8.0
- MindSpore==2.6.0
- mpirun (Open MPI) >= 4.1.2

### 环境安装
```bash
pip install -r requirements.txt
export PYTHONPATH=<mindsponge src路径>:${PYTHONPATH}
```

## 2 运行

### 2.1 Multimer多卡推理

```bash
bash infer_main_parallel.sh  0,1,2,3,4,5,6,7 8064 "./5JDS.pkl;;./step_8000.ckpt;1;1"
```

参数说明：
1. 0,1,2,3,4,5,6,7 代表任意device_id
2. 8064 代表序列长度
3. "./5JDS.pkl;;./step_8000.ckpt;1;1" 字符串包括五个参数输入，分别是raw_feat、restr（可能为空，分号连续）、ckpt_path、iter和num_recycle。例如上述字符串代表的含义如下：
    - raw_feat="./5JDS.pkl"
    - restr="None"
    - ckpt_path="./step_8000.ckpt"
    - iter=1
    - num_cycle=1


### 2.2 Grasp_7R94_多卡推理

```bash
# 由于7R94.pkl对应序列3700+，因此padding至4096.
bash infer_main_parallel.sh  0,1,2,3,4,5,6,7 4096 "./features.pkl;./restr_5perc.pkl;step_14000.ckpt;5;20"
```

参数说明：
1. 0,1,2,3,4,5,6,7 代表任意device_id
2. 4096 代表序列长度
3. "./features.pkl;./restr_5perc.pkl;step_14000.ckpt;5;20"字符串包括五个参数输入，分别是raw_feat、restr（可能为空，分号连续）、ckpt_path、iter和num_recycle。例如上述字符串代表的含义如下：
    - raw_feat="./features.pkl"
    - restr="./restr_5perc.pkl"
    - ckpt_path="./step_14000.ckpt"
    - iter=5
    - num_cycle=20

### 2.3 结果输出
结果日志保存在./output文件夹infer.log（可在infer_main_parallel.sh中更改），pdb文件路径保存在./compare_with_parallel文件夹中。
