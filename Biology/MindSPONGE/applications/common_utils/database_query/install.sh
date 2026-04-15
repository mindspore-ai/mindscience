#!/bin/sh

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