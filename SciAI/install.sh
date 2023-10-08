#!/bin/bash

usage()
{
  echo "Usage:"
  echo "bash install.sh [-f]"
  echo "Options:"
  echo "    -f force uninstall"
}

checkopts()
{
  FORCE="off"
  # Process the options
  while getopts 'f' opt
  do
    OPTARG=$(echo "${OPTARG}" | tr '[:upper:]' '[:lower:]')
    case "${opt}" in
        f)
            FORCE="on"
            ;;
        *)
            echo "Unknown option ${opt}"
            usage
            exit 1
    esac
  done
}

uninstall()
{
  DEVICE=("-gpu" "-ascend" "")
  PACKAGE=("mindelec" "mindsponge" "mindflow")

  for p in "${PACKAGE[@]}"
  do
    for d in "${DEVICE[@]}"
    do
      if [[ "X$FORCE" == "Xon" ]]; then
        echo Y | pip uninstall "$p""$d"
      else
        pip uninstall "$p""$d"
      fi
    done
  done

  echo Y | pip uninstall sciai
}

echo "---------------- SciAI: install starts ----------------"
checkopts "$@"
uninstall
pip install output/sciai*.whl
if [[ "X$?" == "X1" ]]; then
  pip install sciai -i https://pypi.tuna.tsinghua.edu.cn/simple
fi
echo "---------------- SciAI: install ends   ----------------"
