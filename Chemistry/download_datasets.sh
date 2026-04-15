
mkdir -p "datasets"
wget -r -np -nH --cut-dirs=2 $WGET_OPTS -P "datasets" \
  https://download-mindspore.osinfra.cn/mindscience/mindchemistry/
