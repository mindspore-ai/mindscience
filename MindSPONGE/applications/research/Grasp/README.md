# Multimer多卡并行推理

## 1 环境依赖

### 1.1 固件驱动及CANN包版本

```bash
# cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg
runtime_running_version=[7.3.0.1.231:8.0.RC2]
compiler_running_version=[7.3.0.1.231:8.0.RC2]
hccl_running_version=[7.3.0.1.231:8.0.RC2]
opp_running_version=[7.3.0.1.231:8.0.RC2]
toolkit_running_version=[7.3.0.1.231:8.0.RC2]
aoe_running_version=[7.3.0.1.231:8.0.RC2]
ncs_running_version=[7.3.0.1.231:8.0.RC2]
opp_kernel_running_version=[7.3.0.1.231:8.0.RC2]
toolkit_upgrade_version=[7.3.0.1.231:8.0.RC2]
aoe_upgrade_version=[7.3.0.1.231:8.0.RC2]
ncs_upgrade_version=[7.3.0.1.231:8.0.RC2]
opp_kernel_upgrade_version=[7.3.0.1.231:8.0.RC2]
opp_upgrade_version=[7.3.0.1.231:8.0.RC2]
runtime_upgrade_version=[7.3.0.1.231:8.0.RC2]
compiler_upgrade_version=[7.3.0.1.231:8.0.RC2]
hccl_upgrade_version=[7.3.0.1.231:8.0.RC2]
runtime_installed_version=[7.0.0.5.242:7.0.RC1][7.1.0.3.220:7.0.0][7.3.0.1.231:8.0.RC2]
compiler_installed_version=[7.0.0.5.242:7.0.RC1][7.1.0.3.220:7.0.0][7.3.0.1.231:8.0.RC2]
opp_installed_version=[7.0.0.5.242:7.0.RC1][7.1.0.3.220:7.0.0][7.3.0.1.231:8.0.RC2]
toolkit_installed_version=[7.0.0.5.242:7.0.RC1][7.1.0.3.220:7.0.0][7.3.0.1.231:8.0.RC2]
aoe_installed_version=[7.0.0.5.242:7.0.RC1][7.1.0.3.220:7.0.0][7.3.0.1.231:8.0.RC2]
ncs_installed_version=[7.0.0.5.242:7.0.RC1][7.1.0.3.220:7.0.0][7.3.0.1.231:8.0.RC2]
opp_kernel_installed_version=[7.2.T7.0.B121:8.0.RC1.alpha002][7.3.0.1.231:8.0.RC2]
hccl_installed_version=[7.3.0.1.231:8.0.RC2]

```

### 1.2 conda环境依赖

```bash
# source activate python310 && pip list
absl-py==2.1.0
aiohappyeyeballs==2.4.4
aiohttp==3.11.11
aiosignal==1.3.2
anyio==4.8.0
ascendebug @ file:///usr/local/Ascend/ascend-toolkit/8.0.RC2/toolkit/tools/ascendebug-0.1.0-py3-none-any.whl
asttokens @ file:///home/conda/feedstock_root/build_artifacts/asttokens_1733250440834/work
astunparse @ file:///home/conda/feedstock_root/build_artifacts/astunparse_1736248061654/work
async-timeout==5.0.1
attrs==25.1.0
auto-tune @ file:///root/selfgz130520532488/compiler/lib64/auto_tune-0.1.0-py3-none-any.whl
bio==1.7.1
biopython==1.81
biothings_client==0.4.1
biotite==0.40.0
Bottleneck @ file:///croot/bottleneck_1731058648584/work
certifi==2024.12.14
charset-normalizer==3.4.1
click==8.1.8
cloudpickle==3.1.1
contourpy==1.3.1
cycler==0.12.1
dataclasses==0.6
dataflow @ file:///root/selfgz130520532488/compiler/lib64/dataflow-0.0.1-py3-none-any.whl
datasets==2.18.0
decorator==5.1.1
descriptastorus==2.6.1
dill==0.3.8
exceptiongroup==1.2.2
filelock==3.17.0
fonttools==4.56.0
frozenlist==1.5.0
fsspec==2024.2.0
ftfy==6.3.1
glob2==0.7
gprofiler-official==1.0.0
h11==0.14.0
h5py==3.12.1
hccl @ file:///root/selfgz132073717241/hccl/lib64/hccl-0.1.0-py3-none-any.whl
hccl-parser @ file:///usr/local/Ascend/ascend-toolkit/8.0.RC2/toolkit/tools/hccl_parser-0.1-py3-none-any.whl
httpcore==1.0.7
httpx==0.28.1
huggingface-hub==0.27.1
idna==3.10
jieba==0.42.1
Jinja2==3.1.5
joblib==1.4.2
kiwisolver==1.4.8
llm-datadist @ file:///root/selfgz130520532488/compiler/lib64/llm_datadist-0.0.1-py3-none-any.whl
llm-engine @ file:///root/selfgz130520532488/compiler/lib64/llm_engine-0.0.1-py3-none-any.whl
MarkupSafe==3.0.2
matplotlib==3.10.0
mindformers==1.3.2
mindpet==1.0.4
mindsponge_ascend @ file:///nfs/grp/gyqlab/konglp/workspace/multimer_grasp_v11_0430_bac/multimer_grasp_v11_0430_bac/mindscience/MindSPONGE/output/mindsponge_ascend-1.0.0rc2-py3-none-any.whl#sha256=83c220d14ec130a8179def65221617164b66e57cda8d620be46eb80270ba44a9
mindspore @ https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.5.0/MindSpore/unified/aarch64/mindspore-2.5.0-cp310-cp310-linux_aarch64.whl#sha256=1116fd666a059f0480deccd6af04f5e9fe9c019fa88df24a51b0e0fe3c2e55da
ml_dtypes==0.5.1
mpmath==1.3.0
msadvisor @ file:///usr/local/Ascend/ascend-toolkit/8.0.RC2/tools/msadvisor/python/msadvisor-1.0.0-cp37-abi3-linux_aarch64.whl
msgpack==1.1.0
multidict==6.1.0
multiprocess==0.70.16
mygene==3.2.2
networkx==3.4.2
nltk==3.9.1
numexpr @ file:///croot/numexpr_1730215942651/work
numpy==1.23.4
op-compile-tool @ file:///root/selfgz130520532488/compiler/lib64/op_compile_tool-0.1.0-py3-none-any.whl
op-gen @ file:///usr/local/Ascend/ascend-toolkit/8.0.RC2/toolkit/tools/op_gen-0.1-py3-none-any.whl
op-test-frame @ file:///usr/local/Ascend/ascend-toolkit/8.0.RC2/toolkit/tools/op_test_frame-0.1-py3-none-any.whl
opc-tool @ file:///root/selfgz130520532488/compiler/lib64/opc_tool-0.1.0-py3-none-any.whl
opencv-python-headless==4.11.0.86
packaging @ file:///home/conda/feedstock_root/build_artifacts/packaging_1733203243479/work
pandas @ file:///croot/pandas_1732735105235/work/dist/pandas-2.2.3-cp310-cp310-linux_aarch64.whl#sha256=ce019667128a6de8bd8a2994b4bae9691713b9c98906420f2b7dedb0a993963a
pandas-flavor==0.6.0
pillow @ file:///croot/pillow_1734430599218/work
platformdirs==4.3.6
pooch==1.8.2
propcache==0.2.1
protobuf==3.19.1
psutil==6.1.1
pyarrow==12.0.1
pyarrow-hotfix==0.6
pyparsing==3.2.1
python-dateutil @ file:///croot/python-dateutil_1716495745266/work
pytz @ file:///croot/pytz_1713974315080/work
PyYAML==6.0.2
rdkit==2024.9.4
regex==2024.11.6
requests==2.32.3
rouge-chinese==1.0.3
safetensors @ file:///croot/safetensors_1732227620007/work
schedule-search @ file:///root/selfgz130520532488/compiler/lib64/schedule_search-0.1.0-py3-none-any.whl
scikit-learn==1.6.1
scipy==1.13.1
sentencepiece==0.2.0
setproctitle==1.3.4
six @ file:///tmp/build/80754af9/six_1644875935023/work
sniffio==1.3.1
sympy==1.13.3
te @ file:///root/selfgz130520532488/compiler/lib64/te-0.4.0-py3-none-any.whl
threadpoolctl==3.5.0
tiktoken==0.8.0
tokenizers==0.15.0
tornado==6.4.2
tqdm==4.67.1
typing_extensions==4.12.2
tzdata @ file:///croot/python-tzdata_1690578112552/work
urllib3==2.3.0
wcwidth==0.2.13
xarray==2024.7.0
xxhash==3.5.0
yarl==1.18.3

```

#### mpirun版本

```bash
mpirun (Open MPI) 4.1.2
```

## 2 运行

### 2.1 Multimer多卡推理

```bash
bash infer_main_parallel.sh  0,1,2,3,4,5,6,7 8064 "./5JDS.pkl;;./step_8000.ckpt;1;1"
```

1. 0,1,2,3,4,5,6,7 代表任意device_id
2. 8064 代表序列长度
3. "./5JDS.pkl;;./step_8000.ckpt;1;1" 字符串包括五个参数输入，分别是raw_feat、restr（可能为空，分号连续）、ckpt_path、iter和num_recycle。例如上述字符串代表的含义如下：
    1. raw_feat="./5JDS.pkl"
    2. restr="None"
    3. ckpt_path="./step_8000.ckpt"
    4. iter=1
    5. num_cycle=1

```shell
# 结果日志，pdb文件保存在./compare_with_parallel/test4_8064_iter1_recycle10_graph_parallel.pdb
start recycle_cond
recycle 1 diff: 58.07871833571992
end recycle_cond:  True
--------------------start----------------------
--------------------end------------------------
start recycle_cond
recycle 2 diff: 8.910383957501509
end recycle_cond:  True
--------------------start----------------------
[WARNING] PRE_ACT(1020081,fff400e05120,python):2025-03-03-19:29:58.873.106 [mindspore/ccsrc/backend/common/mem_reuse/abstract_dynamic_mem_pool.cc:1036] FreeIdleMemsByEagerFree] Eager free count : 1, free memory : 33637916672, real free : 33598472192, not free : 39444480.
[WARNING] PRE_ACT(1020081,fff400e05120,python):2025-03-03-19:30:19.577.569 [mindspore/ccsrc/backend/common/mem_reuse/abstract_dynamic_mem_pool.cc:1036] FreeIdleMemsByEagerFree] Eager free count : 2, free memory : 23910161920, real free : 23899144192, not free : 11017728.
[WARNING] PRE_ACT(1020081,fff400e05120,python):2025-03-03-19:31:04.954.248 [mindspore/ccsrc/backend/common/mem_reuse/abstract_dynamic_mem_pool.cc:1036] FreeIdleMemsByEagerFree] Eager free count : 3, free memory : 20992928256, real free : 20967325696, not free : 25602560.
--------------------end------------------------
start recycle_cond
recycle 3 diff: 1.9637068183177169
end recycle_cond:  True
--------------------start----------------------
[WARNING] DEVICE(1020081,fff400e05120,python):2025-03-03-19:47:05.102.537 [mindspore/ccsrc/plugin/device/ascend/hal/device/ascend_vmm_adapter.cc:176] MmapDeviceMem] Mapped too much memory, physical_handle_size_ : 29696, max_size : 62277025792.
[WARNING] PRE_ACT(1020081,fff400e05120,python):2025-03-03-19:47:09.315.342 [mindspore/ccsrc/backend/common/mem_reuse/abstract_dynamic_mem_pool.cc:1036] FreeIdleMemsByEagerFree] Eager free count : 4, free memory : 46579264000, real free : 46628077568, not free : 0.
--------------------end------------------------
start recycle_cond
recycle 4 diff: 1.3888285949764172
end recycle_cond:  True
--------------------start----------------------
[WARNING] PRE_ACT(1020081,fff400e05120,python):2025-03-03-20:06:17.195.763 [mindspore/ccsrc/backend/common/mem_reuse/abstract_dynamic_mem_pool.cc:1036] FreeIdleMemsByEagerFree] Eager free count : 5, free memory : 41300996096, real free : 41305505792, not free : 0.
[WARNING] PRE_ACT(1020081,fff400e05120,python):2025-03-03-20:08:13.866.614 [mindspore/ccsrc/backend/common/mem_reuse/abstract_dynamic_mem_pool.cc:1036] FreeIdleMemsByEagerFree] Eager free count : 6, free memory : 24988031488, real free : 24954011648, not free : 34019840.
--------------------end------------------------
start recycle_cond
recycle 5 diff: 10.066165713126406
end recycle_cond:  True
--------------------start----------------------
[WARNING] PRE_ACT(1020081,fff400e05120,python):2025-03-03-20:25:22.280.240 [mindspore/ccsrc/backend/common/mem_reuse/abstract_dynamic_mem_pool.cc:1036] FreeIdleMemsByEagerFree] Eager free count : 7, free memory : 45445552128, real free : 45470449664, not free : 0.
[WARNING] PRE_ACT(1020081,fff400e05120,python):2025-03-03-20:27:17.831.470 [mindspore/ccsrc/backend/common/mem_reuse/abstract_dynamic_mem_pool.cc:1036] FreeIdleMemsByEagerFree] Eager free count : 8, free memory : 24988032512, real free : 24956108800, not free : 31923712.
[WARNING] PRE_ACT(1020081,fff400e05120,python):2025-03-03-20:29:01.096.451 [mindspore/ccsrc/backend/common/mem_reuse/abstract_dynamic_mem_pool.cc:1036] FreeIdleMemsByEagerFree] Eager free count : 9, free memory : 20207269376, real free : 20199768064, not free : 7501312.
[WARNING] PRE_ACT(1020081,fff400e05120,python):2025-03-03-20:29:22.670.183 [mindspore/ccsrc/backend/common/mem_reuse/abstract_dynamic_mem_pool.cc:1036] FreeIdleMemsByEagerFree] Eager free count : 10, free memory : 20810024960, real free : 20791164928, not free : 18860032.
[WARNING] PRE_ACT(1020081,fff400e05120,python):2025-03-03-20:29:24.540.909 [mindspore/ccsrc/backend/common/mem_reuse/abstract_dynamic_mem_pool.cc:1036] FreeIdleMemsByEagerFree] Eager free count : 11, free memory : 16676098048, real free : 16680747008, not free : 0.
[WARNING] PRE_ACT(1020081,fff400e05120,python):2025-03-03-20:29:44.289.865 [mindspore/ccsrc/backend/common/mem_reuse/abstract_dynamic_mem_pool.cc:1036] FreeIdleMemsByEagerFree] Eager free count : 12, free memory : 20810024960, real free : 20803747840, not free : 6277120.
[WARNING] PRE_ACT(1020081,fff400e05120,python):2025-03-03-20:29:46.178.452 [mindspore/ccsrc/backend/common/mem_reuse/abstract_dynamic_mem_pool.cc:1036] FreeIdleMemsByEagerFree] Eager free count : 13, free memory : 16676098048, real free : 16680747008, not free : 0.
[WARNING] PRE_ACT(1020081,fff400e05120,python):2025-03-03-20:30:05.953.177 [mindspore/ccsrc/backend/common/mem_reuse/abstract_dynamic_mem_pool.cc:1036] FreeIdleMemsByEagerFree] Eager free count : 14, free memory : 20810024960, real free : 20799553536, not free : 10471424.
--------------------end------------------------
start recycle_cond
recycle 6 diff: 3.656605440009259
end recycle_cond:  True
--------------------start----------------------
[WARNING] PRE_ACT(1020081,fff400e05120,python):2025-03-03-20:44:36.021.703 [mindspore/ccsrc/backend/common/mem_reuse/abstract_dynamic_mem_pool.cc:1036] FreeIdleMemsByEagerFree] Eager free count : 15, free memory : 43995280384, real free : 44025511936, not free : 0.
[WARNING] PRE_ACT(1020081,fff400e05120,python):2025-03-03-20:46:31.630.084 [mindspore/ccsrc/backend/common/mem_reuse/abstract_dynamic_mem_pool.cc:1036] FreeIdleMemsByEagerFree] Eager free count : 16, free memory : 25020546048, real free : 24991760384, not free : 28785664.
--------------------end------------------------
start recycle_cond
recycle 7 diff: 3.186314201691005
end recycle_cond:  True
--------------------start----------------------
--------------------end------------------------
start recycle_cond
recycle 8 diff: 1.2131272309106085
end recycle_cond:  True
--------------------start----------------------
--------------------end------------------------
start recycle_cond
recycle 9 diff: 0.9297680422422511
end recycle_cond:  True
--------------------start----------------------
--------------------end------------------------
[WARNING] CORE(1020081,ffffa38ab020,python):2025-03-03-22:00:39.424.233 [mindspore/core/include/ir/base_tensor.h:452] data] Try to alloca a large memory, size is:8323596288
 ===================== pdb_path ====================  ./compare_with_parallel/test4_8064_iter1_recycle10_graph_parallel.pdb
Filter Restraints Iteration 1 =============================================
Breakage info ==========
Break number: 0, Max neighbour CA dist: 4.078125

Recall info=============
Stop iteration: RemoveThre,Converged,LastIter
Inference done!
time cost:  13111.61140203476
```

### 2.2 Grasp_7R94_多卡推理

```bash
# 由于7R94.pkl对应序列3700+，因此padding至4096.
bash infer_main_parallel.sh  0,1,2,3,4,5,6,7 4096 "./features.pkl;./restr_5perc.pkl;step_14000.ckpt;5;20"
```

1. 0,1,2,3,4,5,6,7 代表任意device_id
2. 4096 代表序列长度
3. "./features.pkl;./restr_5perc.pkl;step_14000.ckpt;5;20"字符串包括五个参数输入，分别是raw_feat、restr（可能为空，分号连续）、ckpt_path、iter和num_recycle。例如上述字符串代表的含义如下：
    1. raw_feat="./features.pkl"
    2. restr="./restr_5perc.pkl"
    3. ckpt_path="./step_14000.ckpt"
    4. iter=5
    5. num_cycle=20

```shell
# seed=9 结果日志
At least 38 restraints will be used in the final iteration
iter is 5
[WARNING] CORE(2128692,ffff907d5020,python):2025-03-10-10:06:19.623.866 [mindspore/core/include/ir/base_tensor.h:85] NewData] Try to alloca a large memory, size is:4294967296
num_recycle is 20
msa_feat_sum 3841181.6750109335
start recycle_cond
recycle 0 diff: 0.0001
end recycle_cond:  True
--------------------start----------------------
--------------------end------------------------
start recycle_cond
recycle 1 diff: 78.62324050630868
end recycle_cond:  True
--------------------start----------------------
--------------------end------------------------
start recycle_cond
recycle 2 diff: 25.586854637566837
end recycle_cond:  True
--------------------start----------------------
--------------------end------------------------
start recycle_cond
recycle 3 diff: 8.839741836685704
end recycle_cond:  True
--------------------start----------------------
--------------------end------------------------
start recycle_cond
recycle 4 diff: 2.436669909107999
end recycle_cond:  True
--------------------start----------------------
--------------------end------------------------
start recycle_cond
recycle 5 diff: 3.358055246987672
end recycle_cond:  True
--------------------start----------------------
--------------------end------------------------
start recycle_cond
recycle 6 diff: 4.751788874477254
end recycle_cond:  True
--------------------start----------------------
--------------------end------------------------
start recycle_cond
recycle 7 diff: 2.8444712162684724
end recycle_cond:  True
--------------------start----------------------
--------------------end------------------------
start recycle_cond
recycle 8 diff: 1.592084565769719
end recycle_cond:  True
--------------------start----------------------
--------------------end------------------------
start recycle_cond
recycle 9 diff: 0.8363213934548326
end recycle_cond:  True
--------------------start----------------------
--------------------end------------------------
start recycle_cond
recycle 10 diff: 0.6078216719909308
end recycle_cond:  True
--------------------start----------------------
--------------------end------------------------
start recycle_cond
recycle 11 diff: 0.4431440625287018
end recycle_cond:  False
early stop: 11
 ===================== pdb_path ====================  ./compare_with_parallel/test6_4096_iter1_recycle20_graph_parallel.pdb
Filter Restraints Iteration 1 =============================================
inter-residue restraints: 189(189 inter-chain + 0 intra-chain)
Inter-chain restraints
Included! Satisfied! A19/conf84.81/nbdist_avg_ca3.88<==>F477/conf53.87/nbdist_avg_ca4.15/dist_cb18.94, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A20/conf81.57/nbdist_avg_ca3.73<==>F481/conf49.65/nbdist_avg_ca3.65/dist_cb22.52, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A21/conf79.47/nbdist_avg_ca3.42<==>F611/conf62.57/nbdist_avg_ca3.73/dist_cb22.17, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A26/conf77.43/nbdist_avg_ca3.93<==>F477/conf53.87/nbdist_avg_ca4.15/dist_cb21.88, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A43/conf63.87/nbdist_avg_ca3.75<==>C370/conf78.42/nbdist_avg_ca3.93/dist_cb17.47, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A52/conf53.33/nbdist_avg_ca3.96<==>B271/conf74.52/nbdist_avg_ca3.88/dist_cb24.81, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A52/conf53.33/nbdist_avg_ca3.96<==>F466/conf68.76/nbdist_avg_ca3.82/dist_cb15.82, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A52/conf53.33/nbdist_avg_ca3.96<==>F473/conf70.79/nbdist_avg_ca3.82/dist_cb19.02, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A54/conf65.74/nbdist_avg_ca3.89<==>F467/conf66.28/nbdist_avg_ca3.92/dist_cb15.35, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A58/conf72.73/nbdist_avg_ca3.98<==>C293/conf77.78/nbdist_avg_ca3.86/dist_cb18.69, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A58/conf72.73/nbdist_avg_ca3.98<==>F477/conf53.87/nbdist_avg_ca4.15/dist_cb17.69, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A61/conf74.27/nbdist_avg_ca4.30<==>C293/conf77.78/nbdist_avg_ca3.86/dist_cb15.40, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A62/conf53.55/nbdist_avg_ca4.80<==>F486/conf71.07/nbdist_avg_ca3.86/dist_cb15.84, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A62/conf53.55/nbdist_avg_ca4.80<==>F495/conf55.37/nbdist_avg_ca3.75/dist_cb18.12, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A66/conf63.41/nbdist_avg_ca3.98<==>F459/conf64.33/nbdist_avg_ca3.70/dist_cb23.83, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A68/conf76.13/nbdist_avg_ca3.83<==>B322/conf89.01/nbdist_avg_ca3.85/dist_cb18.78, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A69/conf75.64/nbdist_avg_ca3.69<==>B283/conf87.03/nbdist_avg_ca3.81/dist_cb21.70, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A70/conf75.34/nbdist_avg_ca3.69<==>F484/conf63.43/nbdist_avg_ca3.91/dist_cb24.56, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A79/conf82.90/nbdist_avg_ca4.00<==>B291/conf77.93/nbdist_avg_ca3.84/dist_cb24.84, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A79/conf82.90/nbdist_avg_ca4.00<==>B327/conf84.36/nbdist_avg_ca3.76/dist_cb22.64, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A91/conf75.50/nbdist_avg_ca3.83<==>F502/conf66.05/nbdist_avg_ca3.92/dist_cb24.20, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Violated!  A93/conf70.61/nbdist_avg_ca3.85<==>F425/conf74.94/nbdist_avg_ca3.86/dist_cb26.58, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A105/conf83.63/nbdist_avg_ca3.86<==>F611/conf62.57/nbdist_avg_ca3.73/dist_cb24.16, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A132/conf67.23/nbdist_avg_ca4.91<==>F477/conf53.87/nbdist_avg_ca4.15/dist_cb15.59, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A132/conf67.23/nbdist_avg_ca4.91<==>F611/conf62.57/nbdist_avg_ca3.73/dist_cb20.30, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A134/conf87.77/nbdist_avg_ca4.09<==>F477/conf53.87/nbdist_avg_ca4.15/dist_cb20.44, range: 0-25.0, rm_score 0, rm_thre 0.0
Excluded! Violated!  A147/conf83.30/nbdist_avg_ca3.86<==>G410/conf72.35/nbdist_avg_ca3.89/dist_cb106.69, range: 0-25.0, rm_score 76.6875, rm_thre 0.0
Included! Satisfied! A181/conf80.56/nbdist_avg_ca3.93<==>B283/conf87.03/nbdist_avg_ca3.81/dist_cb18.77, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A189/conf86.43/nbdist_avg_ca3.94<==>B267/conf78.14/nbdist_avg_ca3.98/dist_cb22.73, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A190/conf87.19/nbdist_avg_ca3.83<==>B372/conf78.83/nbdist_avg_ca3.83/dist_cb22.34, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A193/conf82.82/nbdist_avg_ca3.83<==>B114/conf77.64/nbdist_avg_ca3.83/dist_cb12.43, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A196/conf75.25/nbdist_avg_ca3.89<==>B186/conf86.98/nbdist_avg_ca3.80/dist_cb18.59, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A196/conf75.25/nbdist_avg_ca3.89<==>B372/conf78.83/nbdist_avg_ca3.83/dist_cb17.92, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A200/conf75.59/nbdist_avg_ca3.82<==>B16/conf81.36/nbdist_avg_ca3.79/dist_cb19.66, range: 0-25.0, rm_score 0, rm_thre 0.0
Excluded! Violated!  A201/conf72.03/nbdist_avg_ca3.83<==>C300/conf85.81/nbdist_avg_ca3.82/dist_cb30.20, range: 0-25.0, rm_score 0.203125, rm_thre 0.0
Included! Satisfied! A204/conf75.94/nbdist_avg_ca3.92<==>B183/conf88.40/nbdist_avg_ca3.79/dist_cb18.00, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A205/conf76.10/nbdist_avg_ca3.87<==>B282/conf87.93/nbdist_avg_ca3.92/dist_cb14.29, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A210/conf82.53/nbdist_avg_ca3.86<==>C320/conf85.81/nbdist_avg_ca3.85/dist_cb24.11, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A214/conf86.91/nbdist_avg_ca3.85<==>B370/conf78.22/nbdist_avg_ca4.05/dist_cb24.73, range: 0-25.0, rm_score 0, rm_thre 0.0
Excluded! Violated!  A217/conf86.06/nbdist_avg_ca3.89<==>F596/conf76.15/nbdist_avg_ca3.94/dist_cb61.78, range: 0-25.0, rm_score 31.78125, rm_thre 0.0
Excluded! Violated!  A233/conf75.74/nbdist_avg_ca3.95<==>E70/conf85.95/nbdist_avg_ca3.74/dist_cb121.50, range: 0-25.0, rm_score 91.5, rm_thre 0.0
Included! Satisfied! A235/conf74.96/nbdist_avg_ca3.81<==>B366/conf82.62/nbdist_avg_ca3.78/dist_cb21.67, range: 0-25.0, rm_score 0, rm_thre 0.0
Excluded! Violated!  A237/conf78.12/nbdist_avg_ca3.82<==>D328/conf79.35/nbdist_avg_ca3.69/dist_cb70.62, range: 0-25.0, rm_score 40.625, rm_thre 0.0
Included! Satisfied! A246/conf71.49/nbdist_avg_ca4.10<==>C280/conf89.75/nbdist_avg_ca3.84/dist_cb15.52, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A246/conf71.49/nbdist_avg_ca4.10<==>C326/conf76.01/nbdist_avg_ca3.85/dist_cb9.06, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A252/conf84.47/nbdist_avg_ca3.88<==>B122/conf86.12/nbdist_avg_ca3.78/dist_cb23.22, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A262/conf91.78/nbdist_avg_ca3.79<==>B284/conf87.21/nbdist_avg_ca3.82/dist_cb22.34, range: 0-25.0, rm_score 0, rm_thre 0.0
Excluded! Violated!  A286/conf86.61/nbdist_avg_ca3.84<==>H500/conf76.92/nbdist_avg_ca3.83/dist_cb81.81, range: 0-25.0, rm_score 51.8125, rm_thre 0.0
Excluded! Violated!  A325/conf88.42/nbdist_avg_ca3.95<==>G503/conf70.46/nbdist_avg_ca3.86/dist_cb113.06, range: 0-25.0, rm_score 83.0625, rm_thre 0.0
Excluded! Violated!  A339/conf83.98/nbdist_avg_ca3.87<==>B263/conf90.52/nbdist_avg_ca3.86/dist_cb45.69, range: 0-25.0, rm_score 15.6875, rm_thre 0.0
Included! Satisfied! A352/conf74.21/nbdist_avg_ca4.02<==>F610/conf65.76/nbdist_avg_ca3.79/dist_cb20.67, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A360/conf75.39/nbdist_avg_ca3.98<==>F612/conf50.48/nbdist_avg_ca3.85/dist_cb22.97, range: 0-25.0, rm_score 0, rm_thre 0.0
Excluded! Violated!  A361/conf83.49/nbdist_avg_ca3.86<==>B126/conf88.13/nbdist_avg_ca3.90/dist_cb77.44, range: 0-25.0, rm_score 47.4375, rm_thre 0.0
Excluded! Violated!  B7/conf58.90/nbdist_avg_ca3.66<==>C210/conf81.56/nbdist_avg_ca3.85/dist_cb76.56, range: 0-25.0, rm_score 46.5625, rm_thre 0.0
Included! Satisfied! B7/conf58.90/nbdist_avg_ca3.66<==>H529/conf80.11/nbdist_avg_ca3.87/dist_cb18.12, range: 0-25.0, rm_score 0, rm_thre 0.0
Excluded! Violated!  B14/conf85.85/nbdist_avg_ca3.69<==>F574/conf76.56/nbdist_avg_ca3.90/dist_cb94.94, range: 0-25.0, rm_score 64.9375, rm_thre 0.0
Excluded! Violated!  B19/conf88.88/nbdist_avg_ca3.73<==>H420/conf74.43/nbdist_avg_ca3.89/dist_cb30.45, range: 0-25.0, rm_score 0.453125, rm_thre 0.0
Included! Satisfied! B45/conf61.83/nbdist_avg_ca3.93<==>D338/conf80.29/nbdist_avg_ca3.88/dist_cb23.92, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! B47/conf51.34/nbdist_avg_ca3.76<==>H458/conf69.88/nbdist_avg_ca3.80/dist_cb12.79, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! B49/conf50.61/nbdist_avg_ca3.65<==>H452/conf53.38/nbdist_avg_ca3.63/dist_cb21.58, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! B52/conf57.78/nbdist_avg_ca3.89<==>H439/conf73.78/nbdist_avg_ca3.73/dist_cb24.02, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! B52/conf57.78/nbdist_avg_ca3.89<==>H467/conf76.38/nbdist_avg_ca4.03/dist_cb11.34, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! B52/conf57.78/nbdist_avg_ca3.89<==>H473/conf73.81/nbdist_avg_ca3.76/dist_cb18.77, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! B54/conf69.12/nbdist_avg_ca3.93<==>C268/conf76.98/nbdist_avg_ca4.00/dist_cb24.80, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! B54/conf69.12/nbdist_avg_ca3.93<==>H478/conf64.28/nbdist_avg_ca3.63/dist_cb14.81, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! B54/conf69.12/nbdist_avg_ca3.93<==>H499/conf72.36/nbdist_avg_ca3.89/dist_cb24.86, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! B60/conf77.50/nbdist_avg_ca3.79<==>D293/conf76.74/nbdist_avg_ca3.88/dist_cb18.08, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! B62/conf73.11/nbdist_avg_ca3.88<==>D141/conf79.74/nbdist_avg_ca3.89/dist_cb23.58, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! B68/conf74.15/nbdist_avg_ca3.98<==>C284/conf84.71/nbdist_avg_ca3.81/dist_cb22.59, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! B68/conf74.15/nbdist_avg_ca3.98<==>C314/conf87.84/nbdist_avg_ca3.79/dist_cb23.83, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! B68/conf74.15/nbdist_avg_ca3.98<==>D290/conf74.07/nbdist_avg_ca4.07/dist_cb13.52, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! B68/conf74.15/nbdist_avg_ca3.98<==>D292/conf78.18/nbdist_avg_ca3.88/dist_cb19.12, range: 0-25.0, rm_score 0, rm_thre 0.0
Excluded! Violated!  B87/conf87.66/nbdist_avg_ca3.89<==>D296/conf80.75/nbdist_avg_ca3.85/dist_cb33.56, range: 0-25.0, rm_score 3.5625, rm_thre 0.0
Excluded! Violated!  B88/conf88.88/nbdist_avg_ca3.79<==>G569/conf78.07/nbdist_avg_ca3.89/dist_cb103.62, range: 0-25.0, rm_score 73.625, rm_thre 0.0
Excluded! Violated!  B91/conf85.51/nbdist_avg_ca3.90<==>E255/conf83.11/nbdist_avg_ca3.86/dist_cb98.44, range: 0-25.0, rm_score 68.4375, rm_thre 0.0
Included! Violated!  B92/conf79.02/nbdist_avg_ca3.79<==>H601/conf74.56/nbdist_avg_ca3.83/dist_cb29.77, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! B102/conf68.78/nbdist_avg_ca4.24<==>H494/conf74.37/nbdist_avg_ca3.74/dist_cb24.25, range: 0-25.0, rm_score 0, rm_thre 0.0
Excluded! Violated!  B120/conf87.94/nbdist_avg_ca3.83<==>E115/conf77.85/nbdist_avg_ca3.87/dist_cb85.06, range: 0-25.0, rm_score 55.0625, rm_thre 0.0
Included! Satisfied! B190/conf89.41/nbdist_avg_ca3.95<==>D290/conf74.07/nbdist_avg_ca4.07/dist_cb23.09, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! B193/conf83.12/nbdist_avg_ca3.95<==>C111/conf82.60/nbdist_avg_ca3.83/dist_cb11.49, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! B193/conf83.12/nbdist_avg_ca3.95<==>C371/conf76.63/nbdist_avg_ca3.75/dist_cb23.30, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Violated!  B200/conf74.76/nbdist_avg_ca3.77<==>C134/conf89.49/nbdist_avg_ca3.88/dist_cb27.59, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! B200/conf74.76/nbdist_avg_ca3.77<==>D295/conf83.60/nbdist_avg_ca3.89/dist_cb21.88, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! B201/conf69.31/nbdist_avg_ca3.92<==>C83/conf83.84/nbdist_avg_ca3.81/dist_cb21.28, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! B204/conf74.54/nbdist_avg_ca3.93<==>C322/conf84.78/nbdist_avg_ca3.91/dist_cb21.41, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! B214/conf86.86/nbdist_avg_ca3.82<==>D328/conf79.35/nbdist_avg_ca3.69/dist_cb23.62, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! B217/conf87.16/nbdist_avg_ca3.87<==>D325/conf75.44/nbdist_avg_ca3.97/dist_cb22.30, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Violated!  B241/conf84.92/nbdist_avg_ca3.79<==>C373/conf73.76/nbdist_avg_ca3.92/dist_cb25.02, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! B251/conf83.94/nbdist_avg_ca3.87<==>D323/conf85.09/nbdist_avg_ca3.93/dist_cb19.97, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! B252/conf86.76/nbdist_avg_ca3.88<==>D325/conf75.44/nbdist_avg_ca3.97/dist_cb21.02, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! B254/conf76.62/nbdist_avg_ca4.01<==>C79/conf77.10/nbdist_avg_ca3.96/dist_cb23.44, range: 0-25.0, rm_score 0, rm_thre 0.0
Excluded! Violated!  B330/conf86.88/nbdist_avg_ca3.81<==>E291/conf75.38/nbdist_avg_ca3.89/dist_cb82.25, range: 0-25.0, rm_score 52.25, rm_thre 0.0
Included! Violated!  B339/conf81.73/nbdist_avg_ca3.96<==>H479/conf61.07/nbdist_avg_ca3.86/dist_cb26.19, range: 0-25.0, rm_score 0, rm_thre 0.0
Excluded! Violated!  B346/conf87.67/nbdist_avg_ca3.82<==>F535/conf85.59/nbdist_avg_ca3.98/dist_cb107.12, range: 0-25.0, rm_score 77.125, rm_thre 0.0
Excluded! Violated!  B360/conf88.00/nbdist_avg_ca3.77<==>F503/conf67.55/nbdist_avg_ca4.02/dist_cb83.88, range: 0-25.0, rm_score 53.875, rm_thre 0.0
Included! Violated!  C8/conf67.08/nbdist_avg_ca3.72<==>G601/conf71.87/nbdist_avg_ca3.81/dist_cb27.84, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C11/conf79.07/nbdist_avg_ca3.97<==>F443/conf56.19/nbdist_avg_ca4.05/dist_cb20.73, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C20/conf84.09/nbdist_avg_ca3.78<==>F455/conf62.60/nbdist_avg_ca3.65/dist_cb22.31, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C24/conf72.05/nbdist_avg_ca3.83<==>G547/conf62.33/nbdist_avg_ca3.91/dist_cb17.23, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C27/conf65.95/nbdist_avg_ca3.94<==>G492/conf62.87/nbdist_avg_ca4.04/dist_cb22.66, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C43/conf64.67/nbdist_avg_ca3.81<==>E296/conf79.67/nbdist_avg_ca4.32/dist_cb24.61, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C45/conf63.25/nbdist_avg_ca3.84<==>E360/conf84.46/nbdist_avg_ca3.74/dist_cb18.95, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C49/conf48.10/nbdist_avg_ca3.64<==>G452/conf52.50/nbdist_avg_ca3.77/dist_cb21.72, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C52/conf59.04/nbdist_avg_ca3.68<==>E351/conf69.40/nbdist_avg_ca3.92/dist_cb20.34, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C54/conf66.82/nbdist_avg_ca3.94<==>G475/conf71.59/nbdist_avg_ca3.88/dist_cb19.28, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C61/conf67.90/nbdist_avg_ca4.17<==>G492/conf62.87/nbdist_avg_ca4.04/dist_cb16.25, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C62/conf57.11/nbdist_avg_ca4.16<==>E325/conf73.76/nbdist_avg_ca3.91/dist_cb24.34, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C62/conf57.11/nbdist_avg_ca4.16<==>G460/conf68.64/nbdist_avg_ca4.05/dist_cb17.66, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C63/conf68.08/nbdist_avg_ca4.49<==>G421/conf68.24/nbdist_avg_ca4.13/dist_cb23.44, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C63/conf68.08/nbdist_avg_ca4.49<==>G500/conf62.57/nbdist_avg_ca4.05/dist_cb22.34, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C68/conf74.76/nbdist_avg_ca3.92<==>E283/conf85.22/nbdist_avg_ca3.80/dist_cb22.33, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C68/conf74.76/nbdist_avg_ca3.92<==>E287/conf75.37/nbdist_avg_ca3.78/dist_cb16.05, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C71/conf83.70/nbdist_avg_ca3.84<==>D285/conf76.41/nbdist_avg_ca3.92/dist_cb16.62, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C73/conf78.34/nbdist_avg_ca3.80<==>D279/conf85.08/nbdist_avg_ca3.93/dist_cb24.09, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C86/conf77.82/nbdist_avg_ca4.12<==>D274/conf79.42/nbdist_avg_ca3.78/dist_cb24.64, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C86/conf77.82/nbdist_avg_ca4.12<==>G501/conf70.39/nbdist_avg_ca4.21/dist_cb24.92, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C90/conf75.43/nbdist_avg_ca4.14<==>G492/conf62.87/nbdist_avg_ca4.04/dist_cb14.47, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C93/conf68.85/nbdist_avg_ca4.14<==>G438/conf73.57/nbdist_avg_ca3.87/dist_cb20.47, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C93/conf68.85/nbdist_avg_ca4.14<==>G465/conf74.53/nbdist_avg_ca3.76/dist_cb22.69, range: 0-25.0, rm_score 0, rm_thre 0.0
Excluded! Violated!  C125/conf88.82/nbdist_avg_ca3.85<==>D346/conf79.41/nbdist_avg_ca3.77/dist_cb66.00, range: 0-25.0, rm_score 36.0, rm_thre 0.0
Included! Satisfied! C129/conf75.29/nbdist_avg_ca4.15<==>G506/conf54.08/nbdist_avg_ca3.95/dist_cb24.53, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C196/conf77.05/nbdist_avg_ca3.91<==>D141/conf79.74/nbdist_avg_ca3.89/dist_cb19.44, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C196/conf77.05/nbdist_avg_ca3.91<==>D370/conf78.90/nbdist_avg_ca4.03/dist_cb19.08, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C196/conf77.05/nbdist_avg_ca3.91<==>E281/conf86.42/nbdist_avg_ca3.93/dist_cb24.95, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C200/conf75.56/nbdist_avg_ca3.79<==>E283/conf85.22/nbdist_avg_ca3.80/dist_cb22.61, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C201/conf73.35/nbdist_avg_ca3.80<==>D192/conf86.92/nbdist_avg_ca3.78/dist_cb23.95, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C202/conf78.49/nbdist_avg_ca3.92<==>D305/conf86.34/nbdist_avg_ca3.81/dist_cb23.73, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C204/conf75.52/nbdist_avg_ca3.94<==>D305/conf86.34/nbdist_avg_ca3.81/dist_cb22.05, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C212/conf87.61/nbdist_avg_ca3.89<==>D273/conf76.92/nbdist_avg_ca3.67/dist_cb19.06, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C236/conf77.65/nbdist_avg_ca3.81<==>E326/conf73.64/nbdist_avg_ca3.87/dist_cb23.20, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C237/conf74.30/nbdist_avg_ca3.70<==>D115/conf77.15/nbdist_avg_ca3.93/dist_cb19.88, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C241/conf82.67/nbdist_avg_ca3.77<==>D77/conf78.19/nbdist_avg_ca3.85/dist_cb22.03, range: 0-25.0, rm_score 0, rm_thre 0.0
Excluded! Violated!  C241/conf82.67/nbdist_avg_ca3.77<==>H490/conf74.14/nbdist_avg_ca3.77/dist_cb59.69, range: 0-25.0, rm_score 29.6875, rm_thre 0.0
Included! Satisfied! C249/conf82.26/nbdist_avg_ca3.87<==>E317/conf86.42/nbdist_avg_ca4.20/dist_cb22.06, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C262/conf89.17/nbdist_avg_ca3.84<==>D180/conf81.69/nbdist_avg_ca3.90/dist_cb23.95, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C268/conf76.98/nbdist_avg_ca4.00<==>D281/conf87.23/nbdist_avg_ca3.99/dist_cb24.09, range: 0-25.0, rm_score 0, rm_thre 0.0
Excluded! Violated!  C286/conf77.84/nbdist_avg_ca3.84<==>D139/conf83.41/nbdist_avg_ca3.80/dist_cb47.72, range: 0-25.0, rm_score 17.71875, rm_thre 0.0
Included! Satisfied! C305/conf86.49/nbdist_avg_ca3.86<==>D286/conf78.16/nbdist_avg_ca3.88/dist_cb20.84, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C346/conf78.58/nbdist_avg_ca3.84<==>F442/conf66.81/nbdist_avg_ca4.13/dist_cb22.52, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C346/conf78.58/nbdist_avg_ca3.84<==>F452/conf57.23/nbdist_avg_ca3.63/dist_cb14.21, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! D28/conf79.20/nbdist_avg_ca3.88<==>H458/conf69.88/nbdist_avg_ca3.80/dist_cb18.19, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! D54/conf80.45/nbdist_avg_ca3.87<==>E265/conf88.52/nbdist_avg_ca3.84/dist_cb22.08, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! D70/conf82.33/nbdist_avg_ca3.84<==>E225/conf86.10/nbdist_avg_ca3.92/dist_cb23.97, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! D71/conf83.73/nbdist_avg_ca3.84<==>E267/conf79.80/nbdist_avg_ca3.96/dist_cb20.02, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! D142/conf76.72/nbdist_avg_ca3.84<==>H452/conf53.38/nbdist_avg_ca3.63/dist_cb16.77, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! D145/conf75.30/nbdist_avg_ca3.71<==>H425/conf75.49/nbdist_avg_ca3.99/dist_cb18.91, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Violated!  D182/conf81.66/nbdist_avg_ca3.81<==>E273/conf75.15/nbdist_avg_ca3.73/dist_cb25.05, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! D193/conf83.74/nbdist_avg_ca3.83<==>E109/conf81.77/nbdist_avg_ca3.77/dist_cb17.14, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! D200/conf75.96/nbdist_avg_ca3.78<==>E74/conf80.61/nbdist_avg_ca3.85/dist_cb14.73, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! D200/conf75.96/nbdist_avg_ca3.78<==>E374/conf73.09/nbdist_avg_ca4.02/dist_cb23.95, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! D202/conf76.71/nbdist_avg_ca3.84<==>E280/conf87.67/nbdist_avg_ca3.83/dist_cb21.30, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! D204/conf75.64/nbdist_avg_ca3.87<==>E272/conf74.24/nbdist_avg_ca3.79/dist_cb4.79, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! D210/conf81.57/nbdist_avg_ca3.76<==>E273/conf75.15/nbdist_avg_ca3.73/dist_cb16.61, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! D233/conf76.43/nbdist_avg_ca3.86<==>E364/conf82.98/nbdist_avg_ca3.86/dist_cb17.47, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! D244/conf84.59/nbdist_avg_ca3.93<==>E79/conf79.32/nbdist_avg_ca3.84/dist_cb19.44, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! D250/conf84.12/nbdist_avg_ca3.88<==>E16/conf81.63/nbdist_avg_ca3.83/dist_cb23.56, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! D251/conf82.11/nbdist_avg_ca3.86<==>E371/conf76.85/nbdist_avg_ca3.82/dist_cb24.19, range: 0-25.0, rm_score 0, rm_thre 0.0
Excluded! Violated!  D251/conf82.11/nbdist_avg_ca3.86<==>F428/conf78.91/nbdist_avg_ca3.75/dist_cb99.62, range: 0-25.0, rm_score 69.625, rm_thre 0.0
Included! Satisfied! D262/conf89.81/nbdist_avg_ca3.85<==>E288/conf74.60/nbdist_avg_ca3.76/dist_cb18.12, range: 0-25.0, rm_score 0, rm_thre 0.0
Excluded! Violated!  D265/conf89.34/nbdist_avg_ca3.83<==>E235/conf79.06/nbdist_avg_ca3.91/dist_cb60.03, range: 0-25.0, rm_score 30.03125, rm_thre 0.0
Included! Satisfied! D272/conf72.62/nbdist_avg_ca3.85<==>E281/conf86.42/nbdist_avg_ca3.93/dist_cb22.73, range: 0-25.0, rm_score 0, rm_thre 0.0
Excluded! Violated!  D306/conf87.30/nbdist_avg_ca3.88<==>H588/conf88.44/nbdist_avg_ca3.82/dist_cb69.69, range: 0-25.0, rm_score 39.6875, rm_thre 0.0
Included! Satisfied! D330/conf77.42/nbdist_avg_ca3.94<==>H487/conf76.09/nbdist_avg_ca3.94/dist_cb20.88, range: 0-25.0, rm_score 0, rm_thre 0.0
Excluded! Violated!  D339/conf79.66/nbdist_avg_ca3.87<==>G576/conf66.50/nbdist_avg_ca3.74/dist_cb96.81, range: 0-25.0, rm_score 66.8125, rm_thre 0.0
Included! Satisfied! D346/conf79.41/nbdist_avg_ca3.77<==>H444/conf74.18/nbdist_avg_ca3.84/dist_cb19.41, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! D352/conf66.97/nbdist_avg_ca3.96<==>H490/conf74.14/nbdist_avg_ca3.77/dist_cb24.45, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! D353/conf62.94/nbdist_avg_ca4.10<==>H484/conf69.82/nbdist_avg_ca3.83/dist_cb14.52, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! E7/conf60.42/nbdist_avg_ca3.71<==>G450/conf54.53/nbdist_avg_ca3.64/dist_cb20.50, range: 0-25.0, rm_score 0, rm_thre 0.0
Excluded! Violated!  E52/conf77.93/nbdist_avg_ca3.71<==>H571/conf82.16/nbdist_avg_ca3.83/dist_cb134.25, range: 0-25.0, rm_score 104.25, rm_thre 0.0
Included! Satisfied! E108/conf83.29/nbdist_avg_ca3.79<==>G456/conf61.81/nbdist_avg_ca3.92/dist_cb21.66, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! E132/conf85.96/nbdist_avg_ca3.81<==>G443/conf71.18/nbdist_avg_ca3.85/dist_cb19.56, range: 0-25.0, rm_score 0, rm_thre 0.0
Excluded! Violated!  E143/conf76.83/nbdist_avg_ca3.84<==>G496/conf65.90/nbdist_avg_ca3.99/dist_cb34.03, range: 0-25.0, rm_score 4.03125, rm_thre 0.0
Included! Satisfied! E144/conf74.61/nbdist_avg_ca3.70<==>G459/conf66.24/nbdist_avg_ca3.92/dist_cb16.27, range: 0-25.0, rm_score 0, rm_thre 0.0
Excluded! Violated!  E146/conf72.00/nbdist_avg_ca3.81<==>F465/conf75.67/nbdist_avg_ca3.70/dist_cb73.38, range: 0-25.0, rm_score 43.375, rm_thre 0.0
Included! Satisfied! E147/conf70.71/nbdist_avg_ca3.84<==>G460/conf68.64/nbdist_avg_ca4.05/dist_cb13.85, range: 0-25.0, rm_score 0, rm_thre 0.0
Excluded! Violated!  E217/conf84.46/nbdist_avg_ca3.81<==>G571/conf75.30/nbdist_avg_ca3.84/dist_cb84.31, range: 0-25.0, rm_score 54.3125, rm_thre 0.0
Excluded! Violated!  E230/conf88.39/nbdist_avg_ca3.89<==>F613/conf64.72/nbdist_avg_ca3.94/dist_cb113.06, range: 0-25.0, rm_score 83.0625, rm_thre 0.0
Excluded! Violated!  E237/conf78.58/nbdist_avg_ca3.80<==>F502/conf66.05/nbdist_avg_ca3.92/dist_cb110.81, range: 0-25.0, rm_score 80.8125, rm_thre 0.0
Excluded! Violated!  E273/conf75.15/nbdist_avg_ca3.73<==>H431/conf83.45/nbdist_avg_ca3.83/dist_cb70.62, range: 0-25.0, rm_score 40.625, rm_thre 0.0
Excluded! Violated!  E278/conf85.06/nbdist_avg_ca3.87<==>G613/conf70.01/nbdist_avg_ca3.71/dist_cb57.28, range: 0-25.0, rm_score 27.28125, rm_thre 0.0
Excluded! Violated!  E288/conf74.60/nbdist_avg_ca3.76<==>F611/conf62.57/nbdist_avg_ca3.73/dist_cb87.44, range: 0-25.0, rm_score 57.4375, rm_thre 0.0
Included! Satisfied! E294/conf72.93/nbdist_avg_ca4.52<==>G450/conf54.53/nbdist_avg_ca3.64/dist_cb23.55, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! E297/conf66.15/nbdist_avg_ca4.30<==>G420/conf64.45/nbdist_avg_ca4.00/dist_cb20.59, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Violated!  E316/conf83.99/nbdist_avg_ca4.24<==>G450/conf54.53/nbdist_avg_ca3.64/dist_cb28.62, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! E330/conf73.99/nbdist_avg_ca4.05<==>G457/conf65.40/nbdist_avg_ca3.92/dist_cb19.36, range: 0-25.0, rm_score 0, rm_thre 0.0
Excluded! Violated!  E338/conf80.62/nbdist_avg_ca3.86<==>F611/conf62.57/nbdist_avg_ca3.73/dist_cb111.31, range: 0-25.0, rm_score 81.3125, rm_thre 0.0
Included! Satisfied! E353/conf65.33/nbdist_avg_ca4.11<==>G472/conf73.60/nbdist_avg_ca3.80/dist_cb14.32, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! E353/conf65.33/nbdist_avg_ca4.11<==>G494/conf69.50/nbdist_avg_ca4.03/dist_cb21.59, range: 0-25.0, rm_score 0, rm_thre 0.0
Excluded! Violated!  F499/conf71.74/nbdist_avg_ca3.83<==>G578/conf66.45/nbdist_avg_ca3.66/dist_cb43.22, range: 0-25.0, rm_score 13.21875, rm_thre 0.0
>>>>> Total 189: 152 included, 144 satisfied
Breakage info ==========
Break number: 2, Max neighbour CA dist: 5.6640625

Recall info=============
interchain (w 1): recall 0.7619047618644494, recall weighted by confidence: 0.7512976084061713
[WARNING] CORE(2128692,ffff907d5020,python):2025-03-10-11:16:17.647.697 [mindspore/core/include/ir/base_tensor.h:85] NewData] Try to alloca a large memory, size is:4294967296
num_recycle is 20
start recycle_cond
recycle 0 diff: 0.0001
end recycle_cond:  True
--------------------start----------------------
--------------------end------------------------
start recycle_cond
recycle 1 diff: 84.20224933251642
end recycle_cond:  True
--------------------start----------------------
--------------------end------------------------
start recycle_cond
recycle 2 diff: 8.531760285440317
end recycle_cond:  True
--------------------start----------------------
--------------------end------------------------
start recycle_cond
recycle 3 diff: 2.998889868861365
end recycle_cond:  True
--------------------start----------------------
--------------------end------------------------
start recycle_cond
recycle 4 diff: 1.8990718744742177
end recycle_cond:  True
--------------------start----------------------
--------------------end------------------------
start recycle_cond
recycle 5 diff: 1.5802629971343825
end recycle_cond:  True
--------------------start----------------------
--------------------end------------------------
start recycle_cond
recycle 6 diff: 1.2450133119931146
end recycle_cond:  True
--------------------start----------------------
--------------------end------------------------
start recycle_cond
recycle 7 diff: 1.0525802643577602
end recycle_cond:  True
--------------------start----------------------
--------------------end------------------------
start recycle_cond
recycle 8 diff: 1.0274764101442193
end recycle_cond:  True
--------------------start----------------------
--------------------end------------------------
start recycle_cond
recycle 9 diff: 0.9284488014707725
end recycle_cond:  True
--------------------start----------------------
--------------------end------------------------
start recycle_cond
recycle 10 diff: 0.5978580326863305
end recycle_cond:  True
--------------------start----------------------
--------------------end------------------------
start recycle_cond
recycle 11 diff: 0.47195285231080886
end recycle_cond:  False
early stop: 11
 ===================== pdb_path ====================  ./compare_with_parallel/test6_4096_iter2_recycle20_graph_parallel.pdb
Filter Restraints Iteration 2 =============================================
inter-residue restraints: 152(152 inter-chain + 0 intra-chain)
Inter-chain restraints
Included! Satisfied! A19/conf86.28/nbdist_avg_ca3.83<==>F477/conf57.81/nbdist_avg_ca3.99/dist_cb18.77, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A20/conf82.55/nbdist_avg_ca3.72<==>F481/conf53.03/nbdist_avg_ca3.64/dist_cb22.64, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A21/conf84.59/nbdist_avg_ca3.51<==>F611/conf69.38/nbdist_avg_ca3.91/dist_cb22.17, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A26/conf78.18/nbdist_avg_ca3.88<==>F477/conf57.81/nbdist_avg_ca3.99/dist_cb22.12, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A43/conf67.06/nbdist_avg_ca3.73<==>C370/conf79.39/nbdist_avg_ca3.91/dist_cb17.56, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A52/conf59.21/nbdist_avg_ca3.85<==>B271/conf75.23/nbdist_avg_ca3.85/dist_cb24.14, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A52/conf59.21/nbdist_avg_ca3.85<==>F466/conf74.17/nbdist_avg_ca3.83/dist_cb15.97, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A52/conf59.21/nbdist_avg_ca3.85<==>F473/conf72.32/nbdist_avg_ca3.80/dist_cb18.81, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A54/conf72.29/nbdist_avg_ca3.86<==>F467/conf73.62/nbdist_avg_ca3.91/dist_cb15.64, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A58/conf75.01/nbdist_avg_ca3.84<==>C293/conf80.21/nbdist_avg_ca3.84/dist_cb18.14, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A58/conf75.01/nbdist_avg_ca3.84<==>F477/conf57.81/nbdist_avg_ca3.99/dist_cb18.09, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A61/conf74.92/nbdist_avg_ca4.04<==>C293/conf80.21/nbdist_avg_ca3.84/dist_cb15.23, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A62/conf66.72/nbdist_avg_ca4.37<==>F486/conf73.79/nbdist_avg_ca3.82/dist_cb16.31, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A62/conf66.72/nbdist_avg_ca4.37<==>F495/conf68.56/nbdist_avg_ca3.74/dist_cb18.72, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A66/conf69.73/nbdist_avg_ca3.96<==>F459/conf71.05/nbdist_avg_ca3.74/dist_cb23.50, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A68/conf77.25/nbdist_avg_ca3.81<==>B322/conf90.03/nbdist_avg_ca3.82/dist_cb18.75, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A69/conf76.07/nbdist_avg_ca3.71<==>B283/conf87.63/nbdist_avg_ca3.79/dist_cb21.59, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A70/conf77.38/nbdist_avg_ca3.70<==>F484/conf67.76/nbdist_avg_ca3.84/dist_cb24.30, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A79/conf84.06/nbdist_avg_ca3.96<==>B291/conf80.31/nbdist_avg_ca3.83/dist_cb24.66, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A79/conf84.06/nbdist_avg_ca3.96<==>B327/conf84.33/nbdist_avg_ca3.74/dist_cb22.44, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A91/conf79.54/nbdist_avg_ca3.83<==>F502/conf73.90/nbdist_avg_ca3.73/dist_cb24.53, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Violated!  A93/conf72.48/nbdist_avg_ca3.84<==>F425/conf74.86/nbdist_avg_ca3.90/dist_cb26.77, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A105/conf86.35/nbdist_avg_ca3.83<==>F611/conf69.38/nbdist_avg_ca3.91/dist_cb24.52, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A132/conf70.46/nbdist_avg_ca4.41<==>F477/conf57.81/nbdist_avg_ca3.99/dist_cb16.91, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A132/conf70.46/nbdist_avg_ca4.41<==>F611/conf69.38/nbdist_avg_ca3.91/dist_cb21.97, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A134/conf89.13/nbdist_avg_ca4.00<==>F477/conf57.81/nbdist_avg_ca3.99/dist_cb20.80, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A181/conf81.97/nbdist_avg_ca3.93<==>B283/conf87.63/nbdist_avg_ca3.79/dist_cb18.75, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A189/conf87.89/nbdist_avg_ca3.94<==>B267/conf79.29/nbdist_avg_ca3.95/dist_cb22.73, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A190/conf87.83/nbdist_avg_ca3.85<==>B372/conf80.62/nbdist_avg_ca3.80/dist_cb22.34, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A193/conf84.63/nbdist_avg_ca3.83<==>B114/conf79.07/nbdist_avg_ca3.85/dist_cb12.41, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A196/conf77.08/nbdist_avg_ca3.89<==>B186/conf87.59/nbdist_avg_ca3.79/dist_cb18.64, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A196/conf77.08/nbdist_avg_ca3.89<==>B372/conf80.62/nbdist_avg_ca3.80/dist_cb17.86, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A200/conf75.14/nbdist_avg_ca3.79<==>B16/conf83.22/nbdist_avg_ca3.77/dist_cb19.58, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A204/conf77.23/nbdist_avg_ca3.87<==>B183/conf88.64/nbdist_avg_ca3.79/dist_cb17.84, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A205/conf76.98/nbdist_avg_ca3.83<==>B282/conf88.75/nbdist_avg_ca3.89/dist_cb14.27, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A210/conf84.59/nbdist_avg_ca3.82<==>C320/conf86.74/nbdist_avg_ca3.86/dist_cb24.06, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A214/conf87.59/nbdist_avg_ca3.84<==>B370/conf79.08/nbdist_avg_ca4.02/dist_cb24.67, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A235/conf76.60/nbdist_avg_ca3.83<==>B366/conf84.70/nbdist_avg_ca3.79/dist_cb21.75, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A246/conf72.93/nbdist_avg_ca4.05<==>C280/conf90.43/nbdist_avg_ca3.84/dist_cb15.36, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A246/conf72.93/nbdist_avg_ca4.05<==>C326/conf77.27/nbdist_avg_ca3.83/dist_cb9.06, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A252/conf84.81/nbdist_avg_ca3.88<==>B122/conf87.07/nbdist_avg_ca3.77/dist_cb23.12, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A262/conf92.31/nbdist_avg_ca3.77<==>B284/conf88.17/nbdist_avg_ca3.83/dist_cb22.41, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A352/conf75.80/nbdist_avg_ca3.99<==>F610/conf72.68/nbdist_avg_ca3.87/dist_cb21.14, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! A360/conf81.19/nbdist_avg_ca3.91<==>F612/conf54.84/nbdist_avg_ca3.95/dist_cb23.89, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! B7/conf59.60/nbdist_avg_ca3.64<==>H529/conf82.94/nbdist_avg_ca3.83/dist_cb18.05, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! B45/conf65.56/nbdist_avg_ca3.90<==>D338/conf83.64/nbdist_avg_ca3.89/dist_cb23.75, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! B47/conf55.95/nbdist_avg_ca3.70<==>H458/conf71.92/nbdist_avg_ca3.79/dist_cb12.81, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! B49/conf54.07/nbdist_avg_ca3.64<==>H452/conf62.02/nbdist_avg_ca3.77/dist_cb21.75, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! B52/conf60.96/nbdist_avg_ca3.77<==>H439/conf74.43/nbdist_avg_ca3.71/dist_cb24.22, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! B52/conf60.96/nbdist_avg_ca3.77<==>H467/conf78.14/nbdist_avg_ca3.96/dist_cb11.59, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! B52/conf60.96/nbdist_avg_ca3.77<==>H473/conf74.49/nbdist_avg_ca3.76/dist_cb18.86, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! B54/conf73.85/nbdist_avg_ca3.90<==>C268/conf77.85/nbdist_avg_ca4.00/dist_cb24.36, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! B54/conf73.85/nbdist_avg_ca3.90<==>H478/conf67.62/nbdist_avg_ca3.67/dist_cb15.02, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Violated!  B54/conf73.85/nbdist_avg_ca3.90<==>H499/conf73.71/nbdist_avg_ca3.89/dist_cb25.16, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! B60/conf79.74/nbdist_avg_ca3.78<==>D293/conf79.23/nbdist_avg_ca3.86/dist_cb18.30, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! B62/conf74.30/nbdist_avg_ca3.88<==>D141/conf81.63/nbdist_avg_ca3.87/dist_cb23.36, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! B68/conf76.22/nbdist_avg_ca3.96<==>C284/conf85.97/nbdist_avg_ca3.82/dist_cb22.64, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! B68/conf76.22/nbdist_avg_ca3.96<==>C314/conf89.20/nbdist_avg_ca3.79/dist_cb23.89, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! B68/conf76.22/nbdist_avg_ca3.96<==>D290/conf74.48/nbdist_avg_ca4.03/dist_cb13.49, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! B68/conf76.22/nbdist_avg_ca3.96<==>D292/conf80.43/nbdist_avg_ca3.88/dist_cb19.09, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Violated!  B92/conf81.87/nbdist_avg_ca3.80<==>H601/conf75.74/nbdist_avg_ca3.83/dist_cb29.83, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! B102/conf70.96/nbdist_avg_ca4.11<==>H494/conf75.11/nbdist_avg_ca3.76/dist_cb24.59, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! B190/conf90.17/nbdist_avg_ca3.94<==>D290/conf74.48/nbdist_avg_ca4.03/dist_cb22.84, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! B193/conf83.56/nbdist_avg_ca3.93<==>C111/conf83.57/nbdist_avg_ca3.83/dist_cb11.48, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! B193/conf83.56/nbdist_avg_ca3.93<==>C371/conf76.83/nbdist_avg_ca3.74/dist_cb23.20, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Violated!  B200/conf75.81/nbdist_avg_ca3.77<==>C134/conf90.83/nbdist_avg_ca3.87/dist_cb27.52, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! B200/conf75.81/nbdist_avg_ca3.77<==>D295/conf86.05/nbdist_avg_ca3.90/dist_cb21.72, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! B201/conf70.68/nbdist_avg_ca3.89<==>C83/conf85.11/nbdist_avg_ca3.81/dist_cb21.27, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! B204/conf77.01/nbdist_avg_ca3.91<==>C322/conf86.73/nbdist_avg_ca3.89/dist_cb21.52, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! B214/conf87.11/nbdist_avg_ca3.82<==>D328/conf80.88/nbdist_avg_ca3.67/dist_cb23.23, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! B217/conf88.48/nbdist_avg_ca3.88<==>D325/conf77.35/nbdist_avg_ca3.95/dist_cb22.06, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! B241/conf86.01/nbdist_avg_ca3.78<==>C373/conf74.26/nbdist_avg_ca3.92/dist_cb24.91, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! B251/conf85.96/nbdist_avg_ca3.86<==>D323/conf86.38/nbdist_avg_ca3.92/dist_cb19.73, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! B252/conf87.41/nbdist_avg_ca3.88<==>D325/conf77.35/nbdist_avg_ca3.95/dist_cb20.77, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! B254/conf77.75/nbdist_avg_ca4.01<==>C79/conf78.29/nbdist_avg_ca3.95/dist_cb23.47, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Violated!  B339/conf83.38/nbdist_avg_ca3.99<==>H479/conf63.39/nbdist_avg_ca3.87/dist_cb25.75, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Violated!  C8/conf68.06/nbdist_avg_ca3.71<==>G601/conf74.52/nbdist_avg_ca3.80/dist_cb27.62, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C11/conf83.60/nbdist_avg_ca3.87<==>F443/conf60.81/nbdist_avg_ca3.98/dist_cb20.36, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C20/conf86.00/nbdist_avg_ca3.77<==>F455/conf68.52/nbdist_avg_ca3.63/dist_cb21.55, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C24/conf74.65/nbdist_avg_ca3.83<==>G547/conf62.74/nbdist_avg_ca4.08/dist_cb16.67, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C27/conf70.68/nbdist_avg_ca3.90<==>G492/conf68.16/nbdist_avg_ca4.04/dist_cb22.00, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C43/conf67.61/nbdist_avg_ca3.79<==>E296/conf83.35/nbdist_avg_ca4.12/dist_cb24.44, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C45/conf67.49/nbdist_avg_ca3.85<==>E360/conf86.23/nbdist_avg_ca3.73/dist_cb19.33, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C49/conf51.65/nbdist_avg_ca3.62<==>G452/conf59.98/nbdist_avg_ca3.87/dist_cb21.77, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C52/conf62.86/nbdist_avg_ca3.67<==>E351/conf72.11/nbdist_avg_ca3.88/dist_cb20.17, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C54/conf72.43/nbdist_avg_ca3.89<==>G475/conf72.58/nbdist_avg_ca3.86/dist_cb19.61, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C61/conf71.83/nbdist_avg_ca3.99<==>G492/conf68.16/nbdist_avg_ca4.04/dist_cb15.87, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C62/conf65.29/nbdist_avg_ca3.96<==>E325/conf73.93/nbdist_avg_ca3.88/dist_cb24.33, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C62/conf65.29/nbdist_avg_ca3.96<==>G460/conf72.79/nbdist_avg_ca3.95/dist_cb17.81, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C63/conf72.46/nbdist_avg_ca4.20<==>G421/conf73.07/nbdist_avg_ca4.02/dist_cb23.77, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C63/conf72.46/nbdist_avg_ca4.20<==>G500/conf68.49/nbdist_avg_ca4.03/dist_cb22.81, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C68/conf77.27/nbdist_avg_ca3.91<==>E283/conf87.53/nbdist_avg_ca3.81/dist_cb22.20, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C68/conf77.27/nbdist_avg_ca3.91<==>E287/conf79.32/nbdist_avg_ca3.82/dist_cb15.92, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C71/conf85.90/nbdist_avg_ca3.83<==>D285/conf78.15/nbdist_avg_ca3.91/dist_cb16.62, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C73/conf79.99/nbdist_avg_ca3.79<==>D279/conf86.25/nbdist_avg_ca3.93/dist_cb23.92, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C86/conf79.75/nbdist_avg_ca4.04<==>D274/conf82.08/nbdist_avg_ca3.77/dist_cb24.30, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Violated!  C86/conf79.75/nbdist_avg_ca4.04<==>G501/conf73.07/nbdist_avg_ca4.08/dist_cb25.22, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C90/conf78.71/nbdist_avg_ca4.05<==>G492/conf68.16/nbdist_avg_ca4.04/dist_cb14.57, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C93/conf72.16/nbdist_avg_ca4.07<==>G438/conf73.78/nbdist_avg_ca3.87/dist_cb20.39, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C93/conf72.16/nbdist_avg_ca4.07<==>G465/conf76.66/nbdist_avg_ca3.76/dist_cb22.47, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Violated!  C129/conf78.20/nbdist_avg_ca4.00<==>G506/conf62.60/nbdist_avg_ca3.89/dist_cb25.58, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C196/conf78.19/nbdist_avg_ca3.90<==>D141/conf81.63/nbdist_avg_ca3.87/dist_cb19.36, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C196/conf78.19/nbdist_avg_ca3.90<==>D370/conf79.54/nbdist_avg_ca3.98/dist_cb18.92, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C196/conf78.19/nbdist_avg_ca3.90<==>E281/conf88.84/nbdist_avg_ca3.92/dist_cb24.83, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C200/conf76.54/nbdist_avg_ca3.77<==>E283/conf87.53/nbdist_avg_ca3.81/dist_cb22.52, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C201/conf75.64/nbdist_avg_ca3.78<==>D192/conf88.39/nbdist_avg_ca3.77/dist_cb24.00, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C202/conf79.56/nbdist_avg_ca3.90<==>D305/conf87.98/nbdist_avg_ca3.82/dist_cb23.77, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C204/conf77.99/nbdist_avg_ca3.92<==>D305/conf87.98/nbdist_avg_ca3.82/dist_cb22.05, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C212/conf89.16/nbdist_avg_ca3.89<==>D273/conf79.04/nbdist_avg_ca3.70/dist_cb19.05, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C236/conf78.92/nbdist_avg_ca3.81<==>E326/conf75.55/nbdist_avg_ca3.86/dist_cb22.77, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C237/conf75.72/nbdist_avg_ca3.72<==>D115/conf78.05/nbdist_avg_ca3.89/dist_cb20.08, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C241/conf85.80/nbdist_avg_ca3.75<==>D77/conf79.81/nbdist_avg_ca3.84/dist_cb22.08, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C249/conf84.81/nbdist_avg_ca3.86<==>E317/conf88.20/nbdist_avg_ca4.15/dist_cb22.00, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C262/conf89.95/nbdist_avg_ca3.83<==>D180/conf83.39/nbdist_avg_ca3.89/dist_cb23.95, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C268/conf77.85/nbdist_avg_ca4.00<==>D281/conf88.62/nbdist_avg_ca4.00/dist_cb24.11, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C305/conf86.88/nbdist_avg_ca3.85<==>D286/conf79.82/nbdist_avg_ca3.88/dist_cb20.88, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C346/conf79.76/nbdist_avg_ca3.80<==>F442/conf71.61/nbdist_avg_ca4.05/dist_cb21.80, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! C346/conf79.76/nbdist_avg_ca3.80<==>F452/conf63.91/nbdist_avg_ca3.73/dist_cb13.95, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! D28/conf82.30/nbdist_avg_ca3.84<==>H458/conf71.92/nbdist_avg_ca3.79/dist_cb17.61, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! D54/conf83.92/nbdist_avg_ca3.85<==>E265/conf90.55/nbdist_avg_ca3.86/dist_cb22.02, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! D70/conf85.32/nbdist_avg_ca3.85<==>E225/conf87.97/nbdist_avg_ca3.89/dist_cb23.98, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! D71/conf85.67/nbdist_avg_ca3.84<==>E267/conf83.93/nbdist_avg_ca3.91/dist_cb20.03, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! D142/conf80.91/nbdist_avg_ca3.82<==>H452/conf62.02/nbdist_avg_ca3.77/dist_cb16.50, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! D145/conf77.18/nbdist_avg_ca3.72<==>H425/conf78.86/nbdist_avg_ca3.93/dist_cb18.34, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Violated!  D182/conf85.30/nbdist_avg_ca3.83<==>E273/conf78.89/nbdist_avg_ca3.71/dist_cb25.16, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! D193/conf85.26/nbdist_avg_ca3.82<==>E109/conf83.14/nbdist_avg_ca3.78/dist_cb17.14, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! D200/conf76.12/nbdist_avg_ca3.78<==>E74/conf81.98/nbdist_avg_ca3.84/dist_cb14.77, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! D200/conf76.12/nbdist_avg_ca3.78<==>E374/conf73.33/nbdist_avg_ca3.98/dist_cb23.94, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! D202/conf78.94/nbdist_avg_ca3.85<==>E280/conf89.41/nbdist_avg_ca3.83/dist_cb21.39, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! D204/conf77.38/nbdist_avg_ca3.86<==>E272/conf75.66/nbdist_avg_ca3.82/dist_cb4.84, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! D210/conf85.32/nbdist_avg_ca3.76<==>E273/conf78.89/nbdist_avg_ca3.71/dist_cb16.75, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! D233/conf77.98/nbdist_avg_ca3.87<==>E364/conf86.15/nbdist_avg_ca3.86/dist_cb17.53, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! D244/conf85.81/nbdist_avg_ca3.93<==>E79/conf81.58/nbdist_avg_ca3.82/dist_cb19.36, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! D250/conf85.28/nbdist_avg_ca3.87<==>E16/conf82.64/nbdist_avg_ca3.83/dist_cb23.64, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! D251/conf84.83/nbdist_avg_ca3.84<==>E371/conf77.88/nbdist_avg_ca3.82/dist_cb24.19, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! D262/conf91.09/nbdist_avg_ca3.86<==>E288/conf76.24/nbdist_avg_ca3.79/dist_cb18.22, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! D272/conf74.58/nbdist_avg_ca3.84<==>E281/conf88.84/nbdist_avg_ca3.92/dist_cb22.69, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! D330/conf82.67/nbdist_avg_ca3.83<==>H487/conf77.20/nbdist_avg_ca3.91/dist_cb20.52, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! D346/conf82.21/nbdist_avg_ca3.75<==>H444/conf75.45/nbdist_avg_ca3.84/dist_cb18.70, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! D352/conf69.56/nbdist_avg_ca3.92<==>H490/conf74.05/nbdist_avg_ca3.82/dist_cb24.47, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! D353/conf65.79/nbdist_avg_ca4.10<==>H484/conf72.63/nbdist_avg_ca3.78/dist_cb14.62, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! E7/conf60.60/nbdist_avg_ca3.71<==>G450/conf61.19/nbdist_avg_ca3.64/dist_cb21.25, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! E108/conf85.17/nbdist_avg_ca3.77<==>G456/conf69.45/nbdist_avg_ca3.89/dist_cb21.52, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! E132/conf87.80/nbdist_avg_ca3.80<==>G443/conf73.68/nbdist_avg_ca3.83/dist_cb19.95, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! E144/conf80.71/nbdist_avg_ca3.75<==>G459/conf70.80/nbdist_avg_ca3.91/dist_cb15.88, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! E147/conf75.02/nbdist_avg_ca3.90<==>G460/conf72.79/nbdist_avg_ca3.95/dist_cb13.56, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! E294/conf73.74/nbdist_avg_ca4.31<==>G450/conf61.19/nbdist_avg_ca3.64/dist_cb23.23, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! E297/conf73.84/nbdist_avg_ca4.12<==>G420/conf71.51/nbdist_avg_ca3.99/dist_cb20.56, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Violated!  E316/conf86.73/nbdist_avg_ca4.13<==>G450/conf61.19/nbdist_avg_ca3.64/dist_cb27.59, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! E330/conf77.72/nbdist_avg_ca3.93<==>G457/conf71.79/nbdist_avg_ca3.91/dist_cb19.08, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! E353/conf66.52/nbdist_avg_ca3.99<==>G472/conf75.47/nbdist_avg_ca3.79/dist_cb14.50, range: 0-25.0, rm_score 0, rm_thre 0.0
Included! Satisfied! E353/conf66.52/nbdist_avg_ca3.99<==>G494/conf73.55/nbdist_avg_ca3.95/dist_cb21.98, range: 0-25.0, rm_score 0, rm_thre 0.0
>>>>> Total 152: 152 included, 142 satisfied
Breakage info ==========
Break number: 0, Max neighbour CA dist: 4.875

Recall info=============
interchain (w 1): recall 0.7513227512829987, recall weighted by confidence: 0.7416862316199593
Stop iteration: Converged
Inference done!
time cost:  6604.073527097702

```

