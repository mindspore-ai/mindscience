#!/bin/sh
echo "***********ma-pre-start*********"
npu-smi info
python -c "import mindspore;mindspore.run_check()"
cd /home/work/user-job-dir/SED || exit
pip install aicc_tools-0.1.7-py3-none-any.whl --ignore-installed
echo "***********ma-pre-end*********"