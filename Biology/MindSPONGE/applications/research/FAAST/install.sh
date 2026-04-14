#!/bin/bash -e

git clone https://github.com/soedinglab/hh-suite.git
mkdir -p hh-suite/build && cd hh-suite/build
cmake -DCMAKE_INSTALL_PREFIX=. ..
make -j 4 && make install
cd ../../
git clone https://github.com/TimoLassmann/kalign.git
cd kalign
mkdir build
cd build
cmake ..
sed -i "2aset(CMAKE_INSTALL_PREFIX ../../)" cmake_install.cmake
make
make test
make install

pip install pandas
pip install pynvml
pip install decorator
pip install tqdm
pip install scikit-learn
pip install pyparsing
pip uninstall --yes urllib3 && pip install urllib3==1.26.14
conda install --yes openmm
conda install --yes -c conda-forge pdbfixer
