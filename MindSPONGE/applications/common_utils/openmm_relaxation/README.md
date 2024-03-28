# OpenMM Relaxation 结构弛豫

AI结构预测方法如MEGA-Fold/AlphaFold2结果只包含碳/氮等重原子的位置信息，缺少氢原子；同时AI方法预测的蛋白质结构可能违反物理化学原理，比如键长键角超出理论值范围等。本工具针对这些缺陷实现基于Amber力场的蛋白质结构弛豫，补全氢原子位置信息的同时使结构更符合物理规律。

## 安装

MindSPONGE参考[安装页面](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE)。

其它依赖库安装参考以下命令：

``` shell
pip install pandas
pip install pynvml
pip install decorator
pip install tqdm
pip install scikit-learn
pip install pyparsing
pip uninstall --yes urllib3 && pip install urllib3==1.26.14
conda install --yes openmm
conda install --yes -c conda-forge pdbfixer
```

OpenMM CPU多核并行能力较差，推荐使用`OMP_NUM_THREADS=5`命令限制实际使用的核数。

## 使用

安装完成后可以运行`test_relax.py`测试，该脚本包含以下测试用例，注意**使用时需将common_utils目录添加到环境变量中**，参考代码：

```python
import sys
sys.path.append("../../common_utils/")  # 将common_utils工具路径添加到环境变量中
from openmm_relaxation.run_relax import run_relax

unrelaxed_pdb_file_path = "../../model_cards/examples/MEGA-Protein/pdb/T1082-D1.pdb"
relaxed_pdb_file_path = "../../model_cards/examples/MEGA-Protein/pdb/T1082-D1_relaxed.pdb"

run_relax(unrelaxed_pdb_file_path, relaxed_pdb_file_path)
```

预期结果：

``` log
Violation of structure after relaxation:  0.0
OpenMM relaxation finished, output pdb file saved at../model_cards/examples/MEGA-Protein/pdb/T1082-D1_relaxed.pdb
```

## 致谢

本工具使用或参考了以下开源工具：

- [AlphaFold2](https://github.com/deepmind/alphafold)
- [Biopython](https://biopython.org)
- [NumPy](https://numpy.org)
- [OpenMM](https://github.com/openmm/openmm)

我们感谢这些开源工具所有的贡献者和维护者！
