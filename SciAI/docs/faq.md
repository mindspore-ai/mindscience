ENGLISH | [简体中文](faq_CN.md)

# SciAI

- [Compilation and Installation related](#Compilation-and-Installation-related)
- [Running with Method 1 (AutoModel advanced API) Related](#Running-with-Method-1-(AutoModel-advanced-API)-Related)
- [Running with Method 2 (source code) Related](#Running-with-Method-2-(source-code)-Related)

## Compilation and Installation related

**Q: After installing `SciAI`, the scientific computing packages in the original environment (such as `MindElec`
, `MindFlow`, `MindSponge`, etc.) are overwritten.**  
A: The `SciAI` basic model library covers Sota models in the industry, some of which are integrated from
other scientific computing suites of `MindScience`. In order to ensure consistent functionalities, it is required to
uninstall or overwrite the original package when installing SciAI.
Optionally, you can install SciAI with source code compilation, and use the following compilation command to avoid
re-installation of
packages that you do not want to uninstall or overwrite:

```shell
bash build.sh -m [pure|mindelec|mindflow|mindsponge] -j8
```

Where `-m [pure|mindelec|mindflow|mindsponge|full]` is optional to compile suites, where `pure` means pure SciAI
compilation (i.e. does not cover any other suite); `full` means all suites compilation.

**Q: Error message  `EOL while scanning string literal` is reported.**  
A: This error is caused by uploading codes from a Windows system to a Linux server and then compiling. Quick fix:
run the following command on `SciAI` directory: `sed -i 's/\r//' version.py`.

**Q: Error message `ModuleNotFoundError: No module named 'mindspore'` is reported.**  
A: This project is based on MindSpore, thereby users need to download and install the matching version of MindSpore to
the current Conda or Python env environment. For details, see [Installation Tutorial](../README.md#installation-guide).

## Running with Method 1 (AutoModel advanced API) Related

**Q: Error message `xxx file not found` is reported, where xxx is a checkpoint file or a dataset file.**  
A: This error is caused by the user deleting the automatically downloaded dataset or checkpoints.
If you are running a model with `AutoModel` advanced API (Running with method 1), you can
add `model.update_config(force_download=True)`before `model.train()` or `model.evaluate()`,
to force automatic download to fix the problem.
If you encounter this problem when running the model in Method 2 (source code), please refer
to [Running with Method 2 (source code) Related](#Running-with-Method-2-(source-code)-Related).

**Q: The terminal prints
`Connecting to download.mindspore.cn (download.mindspore.cn)|xxx.xxx.xxx.xxx|:xxx... failed: Connection timed out`.**  
A: Please check the network connection and proxy settings. If your backend device (GPU/Ascend server) cannot connect to
the network, you can visit
`https://https://download.mindspore.cn/mindscience/SciAI/sciai/model/` Manually download the dataset and upload it to
the corresponding directory in your device.

**Q: Running with method 1 (`AutoModel` advanced API), and the model is from another MindScience suite, there is
no log file after running.**  
A: Some suite models do not contain the logic of log file generation. If the user still wants to generate a log file to
record the training process, please use the following command to redirect to the specified log file.

```shell
python xxx.py >> your_log_file_name.log
```

Where `xxx.py` is the Python script written by the user to call `AutoModel`.

**Q: Error message `'train'|'evluate' function is not supported for model 'xxx'` is reported.**  
A: `train` or `evaluate` is not supported for the specified model. If `train` is not supported, please try `evaluate`;
if `evaluate` is not supported, please try `train` instead.

## Running with Method 2 (source code) Related

**Q: When running a model with method 2 (source code), error message `Module not found error: No module named 'sciai'`
or `ImportError: cannot import name 'xxx' from 'sciai.xxx' (xxx/mindscience/SciAI/sciai/xxx)` is reported.**  
A: This error is caused by not setting `PYTHONPATH`, and you can `cd` to the `SciAI` directory and execute
`source .env` to fix it.

**Q: When running a model with method 2 (source code) and having modified the source code, it is found that the
modified place does not take effect during script execution.**  
A: This problem is caused by the fact that the model is run with source code (method 2) but `PYTHONPATH` is not set,
while the SciAI package has been installed. You can `cd` to the `SciAI` directory and execute `source .env`
to fix it.

**Q: Error message `xxx file not found` is reported, where xxx is a checkpoint file or a dataset file.**  
A: This error is caused by the user accidentally deleting the automatically downloaded dataset or checkpoints.
If you are running the model with method 2 (source code), you can fix it as follows:
When starting `train.py` or `eval.py`, add the parameter `--force_download True` to the run command.
If you encounter this problem when running the model with method 1(AutoModel Advance API), please refer
to [Running with Method 1 (AutoModel advanced API) Related](#Running-with-Method-1-(AutoModel-advanced-API)-Related).
