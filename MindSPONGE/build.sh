#!/bin/bash
# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
set -e
BASEPATH=$(cd "$(dirname $0)"; pwd)
OUTPUT_PATH="${BASEPATH}/output/"
PYTHON=$(which python3)

mk_new_dir() {
    local create_dir="$1"  # the target to make

    if [[ -d "${create_dir}" ]];then
        rm -rf "${create_dir}"
    fi

    mkdir -pv "${create_dir}"
}

write_checksum() {
    cd "$OUTPUT_PATH" || exit
    PACKAGE_LIST=$(ls mindscience_sponge*.whl) || exit
    for PACKAGE_NAME in $PACKAGE_LIST; do
        echo $PACKAGE_NAME
        sha256sum -b "$PACKAGE_NAME" >"$PACKAGE_NAME.sha256"
    done
}

usage()
{
  echo "Usage:"
  echo "bash build.sh [-e gpu|ascend] [-j[n]] [-t on|off]"
  echo "Options:"
  echo "    -e Use gpu or ascend. Currently only support ascend, later will support GPU"
  echo "    -j[n] Set the threads when building (Default: -j8)"
  echo "    -t whether to compile traditional sponge(GPU platform)"
  echo "    -c whether to build Cybertron(A basic Molecular Representation NN toolkit)"
}

# check value of input is 'on' or 'off'
# usage: check_on_off arg_value arg_name
check_on_off()
{
  if [[ "X$1" != "Xon" && "X$1" != "Xoff" ]]; then
    echo "Invalid value $1 for option -$2"
    usage
    exit 1
  fi
}

checkopts()
{
  # Init default values of build options
  ENABLE_D="off"
  ENABLE_GPU="off"
  ENABLE_MD="off"
  ENABLE_CYBERTRON="off"
  THREAD_NUM=8
  # Process the options
  while getopts 'drvj:e:t:c:s:S' opt
  do
    OPTARG=$(echo ${OPTARG} | tr '[A-Z]' '[a-z]')
    case "${opt}" in
        e)
            DEVICE=$OPTARG
            ;;
        j)
            THREAD_NUM=$OPTARG
            ;;
        t)
            ENABLE_MD=$OPTARG
            ;;
        c)
            ENABLE_CYBERTRON=$OPTARG
            ;;
        *)
            echo "Unknown option ${opt}"
            usage
            exit 1
    esac
  done
  if [[ "X$DEVICE" == "Xd" || "X$DEVICE" == "Xascend" ]]; then
    ENABLE_D="on"
  elif [[ "X$DEVICE" == "Xgpu" ]]; then
      ENABLE_GPU="on"
  fi

  if [[ "X$ENABLE_GPU" == "Xoff"  &&  "X$ENABLE_D" == "Xoff" ]]; then
    echo "Error : you should set the backend (gpu | ascend) add -e [gpu|ascend] to you command"
    exit 1
  fi

  if [[ "X$ENABLE_GPU" == "Xoff"  &&  "X$ENABLE_MD" == "Xon" ]]; then
    echo "Error : Traditional Sponge should be compiled on gpu"
    exit 1
  fi
}

build_mindsponge()
{
  echo "---------------- MindSPONGE: build start ----------------"
  mk_new_dir "${BASEPATH}/build/"
  mk_new_dir "${OUTPUT_PATH}"
  if [[ "X$ENABLE_D" = "Xon" ]]; then
    echo "build ascend backend"
    export SPONGE_PACKAGE_NAME=mindscience_sponge_ascend
    CMAKE_FLAG="-DENABLE_D=ON"
  fi
  if [[ "X$ENABLE_CYBERTRON" = "Xon" ]]; then
    echo "build Cybertron"
    cp -r "${BASEPATH}/cybertron/" "${BASEPATH}/build/"
    mv "${BASEPATH}/build/cybertron/setup.py" "${BASEPATH}/build/"
    mv "${BASEPATH}/build/cybertron/requirements.txt" "${BASEPATH}/build/"
    export CYBERTRON_PACKAGE_NAME=mindscience_cybertron
    cd ${BASEPATH}/build/
    ${PYTHON} ./setup.py bdist_wheel
    cd ..
    mv ${BASEPATH}/build/dist/mindscience_cybertron*.whl ${OUTPUT_PATH}
    rm -rf ${BASEPATH}/build/*
  fi
  if [[ "X$ENABLE_GPU" = "Xon" ]]; then
    echo "build gpu backend"
    export SPONGE_PACKAGE_NAME=mindscience_sponge_gpu
    CMAKE_FLAG="-DENABLE_GPU=ON"
    if [[ "X$ENABLE_MD" = "Xon" ]]; then
      CMAKE_FLAG="${CMAKE_FLAG} -DENABLE_MD=ON"
    fi
  fi
  cp -r "${BASEPATH}/mindsponge/python/" "${BASEPATH}/build/mindsponge/"
  cp -r "${BASEPATH}/mindsponge/toolkits/" "${BASEPATH}/build/mindsponge/"
  cp "${BASEPATH}/setup.py" "${BASEPATH}/build/"
  cd "${BASEPATH}/build/"
  echo ${CMAKE_FLAG}
  cmake .. ${CMAKE_FLAG}
  make -j$THREAD_NUM
  ${PYTHON} ./setup.py bdist_wheel
  cd ..
  mv ${BASEPATH}/build/dist/*whl ${OUTPUT_PATH}
  write_checksum
  echo "---------------- MindSPONGE: build end ----------------"
}

checkopts "$@"
build_mindsponge
