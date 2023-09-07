#!/bin/bash
# Copyright 2021 Huawei Technologies Co., Ltd
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

BASEPATH=$(cd "$(dirname "$0")"; pwd)

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
  echo "bash build.sh [-d] [-v] [-e gpu|ascend] [-j[n]] [-m mindflow|mindelec|mindsponge|pure]"
  echo "Options:"
  echo "    -d Debug mode"
  echo "    -e Use gpu or ascend"
  echo "    -r Release mode, default mode"
  echo "    -j[n] Set the threads when building (Default: -j8)"
  echo "    -v Display build command"
  echo "    -d whether to create time in the package"
  echo "    -m include models in the package"
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

compile_models()
{
  if [[ "X$MODEL_TYPE" == "Xmindflow" ]]; then
    THIRD_PACKAGE_NAME=("MindFlow")
    args=(0)
    cp_models
  elif [[ "X$MODEL_TYPE" == "Xmindelec" ]]; then
    THIRD_PACKAGE_NAME=("MindElec")
    args=(1)
    cp_models
  elif [[ "X$MODEL_TYPE" == "Xmindsponge" ]]; then
    THIRD_PACKAGE_NAME=("MindSPONGE")
    args=(2)
  elif [[ "X$MODEL_TYPE" == "Xall" ]]; then
    THIRD_PACKAGE_NAME=("MindFlow" "MindElec" "MindSPONGE")
    args=(0 1 2)
    cp_models
  elif [[ "X$MODEL_TYPE" != "Xpure" ]]; then
    echo "Invalid value $MODEL_TYPE for option -m"
    usage
    exit 1
  fi
  if [[ "X$MODEL_TYPE" != "Xpure" ]]; then
    echo "WARNING: will checkout to different branch and commit, please make sure local changes are stashed..."
    sleep 3
    orig_branch=$(git symbolic-ref --short HEAD)
    i=0
    for var in "${THIRD_PACKAGE_NAME[@]}"
    do
      package_version=$(python -c "import setup; setup.get_version(${args[$i]})")
      IFS=" " read -r -a branch_commit <<< "$(echo "$package_version" | tr ':' ' ')"
      cd "$BASEPATH"/../"$var"
      git checkout "${branch_commit[0]}"
      git checkout "${branch_commit[1]}"
      bash build.sh -e "$DEVICE" -j8
      var_lower=$(echo "$var" | tr '[:upper:]' '[:lower:]')
      echo Y | pip uninstall "$var_lower"
      echo Y | pip uninstall "$var_lower"-"$DEVICE"
      rm -rf build
      cd "$BASEPATH"
      ((i=i+1))
      git checkout $orig_branch
    done
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
  MODEL_TYPE="pure"
  DEVICE="ascend"
  # Process the options
  while getopts 'e:d:rvj:m:s:S' opt
  do
    OPTARG=$(echo "${OPTARG}" | tr '[:upper:]' '[:lower:]')
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
            check_on_off "$OPTARG" S
            ENABLE_GITEE="$OPTARG"
            echo "enable download from gitee"
            ;;

        v)
            VERBOSE="VERBOSE=1"
            ;;
        m)
            MODEL_TYPE=$OPTARG
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
build_sciai()
{
  echo "start build sciai project."

  if [[ "X$ENABLE_DAILY" = "Xon" ]]; then
    names=$(cat ./version.txt)
    time2=$(date "+%Y%m%d")
    for line in $names
    do
      rm -rf ./version.txt
      echo "$line"'.'"$time2" >>./version.txt
      break
    done
  fi

  VERSION=$(cat ./version.txt | tr -d '\r')
  echo "__version__ = '$VERSION'" > ./sciai/version.py
  
  mkdir -pv "${BUILD_PATH}sciai"
  cd "${BUILD_PATH}sciai"

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
  cmake "${CMAKE_ARGS}" ../../.

  if [[ -n "$VERBOSE" ]]; then
    CMAKE_VERBOSE="--verbose"
  fi

  cmake --build . --target package ${CMAKE_VERBOSE} -j"$THREAD_NUM"
  echo "the end, build file written to ${BUILD_PATH}"
}

write_checksum() {
    cd "$OUTPUT_PATH" || exit
    PACKAGE_LIST=$(ls sciai*.whl) || exit
    for PACKAGE_NAME in $PACKAGE_LIST; do
        echo "$PACKAGE_NAME"
        sha256sum -b "$PACKAGE_NAME" >"$PACKAGE_NAME.sha256"
    done
}

cp_models() {
  echo "copy models from other kits..."
  if [[ "X$MODEL_TYPE" == "Xmindelec" ]]; then
    arg=(1)
  elif [[ "X$MODEL_TYPE" == "Xmindflow" ]]; then
    arg=(0)
  else
    arg=(0 1)
  fi
  i=0
  for model_id in "${arg[@]}"
  do
    model_paths=$(python -c "import setup; setup.get_model_path($model_id)")
    IFS=" " read -r -a model_paths_array <<< "$(echo "$model_paths" | tr ',' ' ')"
    for var in "${model_paths_array[@]}"
    do
      IFS=" " read -r -a single_model_path <<< "$(echo "$var" | tr ':' ' ')"
      dir_list[$i]="${single_model_path[0]}"
      cp -r "${single_model_path[1]}" sciai/model/suite_models/"${single_model_path[0]}"
      ((i=i+1))
    done
  done
}

remove_copied_models() {
  echo "remove copied models..."
  for var in "${dir_list[@]}"
  do
    rm -rf sciai/model/"$var"
  done
}

merge_whls() {
  cd "$BASEPATH"/output
  mkdir merge_whls
  cp "$BASEPATH"/../*/output/*.whl merge_whls
  cd merge_whls

  # define suites whl files and corresponding temp directory
  SCIAI_WHL=(sciai*.whl)
  if [[ "X$MODEL_TYPE" == "Xmindflow" ]]; then
    WHL_FILES=("sciai*.whl" "mindflow*.whl")
    TEMP_DIRS=("temp1" "temp2")
  elif [[ "X$MODEL_TYPE" == "Xmindelec" ]]; then
    WHL_FILES=("sciai*.whl" "mindelec*.whl")
    TEMP_DIRS=("temp1" "temp2")
  elif [[ "X$MODEL_TYPE" == "Xmindsponge" ]]; then
    WHL_FILES=("sciai*.whl" "mindsponge*.whl")
    TEMP_DIRS=("temp1" "temp2")
  else
    WHL_FILES=("sciai*.whl" "mindelec*.whl" "mindflow*.whl" "mindsponge*.whl")
    TEMP_DIRS=("temp1" "temp2" "temp3" "temp4")
  fi

  # unzip all .whl files to corresponding temp directories
  for i in "${!WHL_FILES[@]}"; do
    unzip "${WHL_FILES[$i]}" -d "${TEMP_DIRS[$i]}"
  done

  # create merged directory
  MERGED_DIR="merged"
  mkdir "$MERGED_DIR"

  # copy all temp directories into merged directory
  for dir in "${TEMP_DIRS[@]}"; do
    cp -r "$dir"/* "$MERGED_DIR"
  done

  # merge METADATA
  rm -rf merged/*dist-info
  SCIAI_DIST_INFO="temp1/sciai*dist-info"
  SCIAI_DIST_INFO_BASE=$(basename "$SCIAI_DIST_INFO")
  cp -r $SCIAI_DIST_INFO merged

  # copy all dependencies information(if has)
  for dir in "${TEMP_DIRS[@]:1}"; do
    grep "Requires-Dist" $dir/*dist-info/METADATA >> merged/$SCIAI_DIST_INFO_BASE/METADATA
  done

  # merge all files into new .whl
  cd "$MERGED_DIR"
  zip -r "${SCIAI_WHL[0]}" ./*

  echo "Merging process completed."
  mv sciai-*.whl ../../
  cd ../../
  rm -rf merged_whls
}
echo "---------------- SciAI: build start ----------------"
checkopts "$@"
compile_models
build_sciai
mv "${BASEPATH}"/build/package/*whl "${OUTPUT_PATH}"
if [[ "X$MODEL_TYPE" != "Xpure" ]]; then
  merge_whls
fi
write_checksum
cd "$BASEPATH"
echo "---------------- SciAI: build end   ----------------"
