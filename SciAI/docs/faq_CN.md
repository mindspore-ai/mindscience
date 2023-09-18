[ENGLISH](faq.md) | 简体中文

# SciAI

- [编译与安装相关](#编译与安装相关)
- [方式一启动（AutoModel高阶API启动）相关](#方式一启动（AutoModel高阶API启动）相关)
- [方式二启动（源码启动）相关](#方式二启动（源码启动）相关)

## 编译与安装相关

**Q：安装`SciAI`后，原有环境中的科学计算套件（如`MindElec`, `MindFlow`, `MindSponge`等）被覆盖。**  
A：`SciAI`基础模型库涵盖业界Sota模型，其中若干模型集成依赖于`MindScience`其他科学计算套件。为保证功能一致，安装SciAI时要求卸载或覆盖掉原有套件。
选择性地，您可以使用源码编译的方式安装SciAI，在执行编译步骤时使用如下命令，屏蔽掉不想卸载或覆盖掉的套件：

```shell
bash build.sh -m [pure|mindelec|mindflow|mindsponge] -j8
```

其中 `-m [pure|mindelec|mindflow|mindsponge|full]` 为可选择编译的套件，`pure`为编译纯sciai（即不涵盖任何其他套件）; `full`
为编译所有套件。

**Q：报错 `EOL while scanning string literal`。**  
A：该错误由于window上传代码至linux服务器然后编译导致。快速修复：  
在`SciAI`目录下运行如下命令：`sed -i 's/\r//' version.py`

**Q：报错`ModuleNotFoundError: No module named 'mindspore'`。**  
A：本项目基于MindSpore，需下载安装匹配版本的MindSpore至当前Conda或Python env环境。
详见[安装教程](../README_CN.md#确认系统环境信息)。

## 方式一启动（AutoModel高阶API启动）相关

**Q：报错 `xxx file not found`， 其中xxx为checkpoint文件或数据集文件。**  
A：该错误由用户删除模型所对应的自动下载的数据集/checkpoint导致。
如果您正在使用方式一启动模型(`AutoModel`高阶API启动)，可以通过在 `model.train()` 或 `model.evaluate()`
之前加上`model.update_config(force_download=True)` 强制自动下载修复问题。
如果您在使用方式二启动模型（源码启动）遇到此问题，请参考[方式二启动（源码启动）相关](#方式二启动（源码启动）相关)。

**Q：终端打印 `Connecting to download.mindspore.cn  (download.mindspore.cn)|xxx.xxx.xxx.xxx|：xxx... failed：Connection
timed out`。**  
A：请检查网络连接与proxy代理设置。如果您的后端设备（GPU/Ascend服务器）无法连接网络，可访问
`https://https://download.mindspore.cn/mindscience/SciAI/sciai/model/` 手动下载数据集并上传至您的后端设备对应目录下。

**Q：使用方式一启动(AutoModel高阶API），且调用模型为其他MindScience套件模型，遇到运行完没有log文件生成的问题。**  
A：有些套件模型不含log文件生成的逻辑，如果用户希望生成log文件记录训练过程，请用如下命令重定向至指定log文件。

```shell
python xxx.py >> your_log_file_name.log
```

其中`xxx.py`为用户所写的调用`AutoModel`的python脚本。

**Q：报错`'train'|'evluate' function is not supported for model 'xxx'`。**  
A：该模型不支持`train` 或不支持`evaluate`。如果不支持`train`，则请尝试`evaluate`；如果不支持`evaluate`，请尝试`train`。

## 方式二启动（源码启动）相关

**Q：源码运行模型时，报错 `Module not found error: No module named 'sciai'`
或 `ImportError: cannot import name 'xxx' from 'sciai.xxx' (xxx/mindscience/SciAI/sciai/xxx)`。**  
A：该错误由于未设置`PYTHONPATH`引起，可 `cd` 至 `SciAI` 目录下执行 `source .env` 修复。

**Q：使用方式二启动（源码启动）并对源码有修改的情况下，发现修改点在运行中不生效。**  
A：该问题原因为以方式二启动（源码启动）却未设置`PYTHONPATH`, 且环境又安装了SciAI软件包。可 `cd` 至 `SciAI`
目录下执行 `source .env` 修复。详情参考[README.md](../README_CN.md#方式二：源码启动)。

**Q：报错 `xxx file not found`， 其中xxx为checkpoint文件或数据集文件。**  
A：该错误由用户误删自动下载的数据集/checkpoint导致。
如果您在使用方式二启动（源码启动），可通过如下方式修复：
在启动`train.py`或`eval.py`时，在命令中增加参数`--force_download True`。
如果您在使用方式一启动（AutoModel高阶API启动）遇到此问题，请参考[方式一启动（AutoModel高阶API启动）相关](#方式一启动（AutoModel高阶API启动）相关)
。
。
