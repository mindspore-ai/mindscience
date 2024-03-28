# 通用工具 Common Utils

提供生物计算领域常用的通用工具，比如蛋白质结构预测需要的多重序列比对与模板检索，还有基于Amber力场的OpenMM Relaxation，更多工具持续上架中。

## 使用方法

不同工具的依赖安装与使用请参考对应目录下的`README.md`，使用时需将本目录加到python目录中，示例代码：

```python
import sys
sys.path.append("../../common_utils/")  # 将common_utils工具路径添加到环境变量中
from openmm_relaxation.run_relax import run_relax # 结构弛豫
from database_query.protein_feature import RawFeatureGenerator # 多重序列比对与模板检索
```
