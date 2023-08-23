#!/bin/bash
# Copyright 2023 Huawei Technologies Co., Ltd
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

OUTPUT_PATH="${BASEPATH}/output"

mk_new_dir() {
    local create_dir="$1"  # the target to make

    if [[ -d "${create_dir}" ]];then
        rm -rf "${create_dir}"
    fi

    mkdir -pv "${create_dir}"
}

mk_new_dir "${OUTPUT_PATH}"

export BUILD_PATH="${BASEPATH}/build/"

usage()
{
  echo "Usage:"
  echo "bash build.sh [-d] [-v] [-e gpu|ascend] [-j[n]]"
  echo "Options:"
  echo "    -d Debug mode"
  echo "    -e Use gpu or ascend"
  echo "    -r Release mode, default mode"
  echo "    -j[n] Set the threads when building (Default: -j8)"
  echo "    -v Display build command"
  echo "    -d whether to create time in the package"
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
  DEBUG_MODE="off"
  THREAD_NUM=8
  VERBOSE=""
  ENABLE_DAILY="off"
  # Process the options
  while getopts 'e:d:rvj:s:S' opt
  do
    OPTARG=$(echo ${OPTARG} | tr '[A-Z]' '[a-z]')
    case "${opt}" in
        e)
            DEVICE=$OPTARG
            ;;
        d)
            ENABLE_DAILY=$OPTARG
            ;;
        j)
            THREAD_NUM=$OPTARG
            ;;
        r)
            DEBUG_MODE="off"
            ;;
        S)
            check_on_off $OPTARG S
            ENABLE_GITEE="$OPTARG"
            echo "enable download from gitee"
            ;;

        v)
            VERBOSE="VERBOSE=1"
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
}

#Create building path
build_mindearth()
{
  echo "start build mindearth project."

  if [[ "X$ENABLE_DAILY" = "Xon" ]]; then
    names=$(cat ./version.txt)
    time2=$(date "+%Y%m%d")
    for line in $names
    do
      rm -rf ./version.txt
      echo $line'.'$time2 >>./version.txt
      break
    done
  fi
  
  mkdir -pv "${BUILD_PATH}/mindearth"
  cd "${BUILD_PATH}/mindearth"

  CMAKE_ARGS="-DDEBUG_MODE=$DEBUG_MODE -DBUILD_PATH=$BUILD_PATH"

  if [[ "X$ENABLE_GITEE" = "Xon" ]]; then
    CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_GITEE=ON"
  fi

  if [[ "X$ENABLE_D" = "Xon" ]]; then
    CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_D=ON"
  fi

  if [[ "X$ENABLE_GPU" = "Xon" ]]; then
    CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_GPU=ON"
  fi

  echo "${CMAKE_ARGS}"
  cmake ${CMAKE_ARGS} ../../.

  if [[ -n "$VERBOSE" ]]; then
    CMAKE_VERBOSE="--verbose"
  fi

  cmake --build . --target package ${CMAKE_VERBOSE} -j$THREAD_NUM
  echo "the end, build file written to ${BUILD_PATH}"
}

write_checksum() {
    cd "$OUTPUT_PATH" || exit
    PACKAGE_LIST=$(ls mindearth*.whl) || exit
    for PACKAGE_NAME in $PACKAGE_LIST; do
        echo $PACKAGE_NAME
        sha256sum -b "$PACKAGE_NAME" >"$PACKAGE_NAME.sha256"
    done
}

echo "---------------- MindEarth: build start ----------------"
checkopts "$@"
build_mindearth
mv ${BASEPATH}/build/package/*whl ${OUTPUT_PATH}
write_checksum
echo "---------------- MindEarth: build end   ----------------"
