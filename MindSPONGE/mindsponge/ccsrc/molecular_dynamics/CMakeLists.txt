project(sponge_traditional)
cmake_minimum_required(VERSION 3.10.2)
set(backend gpu)
find_package(CUDA)
if(NOT CUDA_FOUND)
message(STATUS "CUDA not found. Project will not be built.")
endif()

set(CMAKE_VERBOSE_MAKEFILEON ON)
set(CUDA_SEPARABLE_COMPILATION ON)

if(NOT CUDA_PATH OR CUDA_PATH STREQUAL "")
  if(DEFINED ENV{CUDA_PATH} AND NOT $ENV{CUDA_PATH} STREQUAL "")
    set(CUDA_PATH $ENV{CUDA_PATH})
  else()
    set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})
  endif()
endif()

if(CUDA_VERSION VERSION_LESS "11.0")
  set(CUDA_NVCC_FLAGS -O3; -std=c++11; -arch=sm_50; -Xcompiler -fPIC; --use_fast_math)
else()
  set(CUDA_NVCC_FLAGS -O3; -std=c++14; -arch=sm_75; -Xcompiler -fPIC; --use_fast_math)
endif()

file(GLOB_RECURSE CURRENT_HEADERS  *.cuh)
file(GLOB_RECURSE CURRENT_SOURCES *.cu)
list(REMOVE_ITEM CURRENT_HEADERS "${PROJECT_SOURCE_DIR}/main.cuh" "${PROJECT_SOURCE_DIR}/main_nopbc.cu"
  "${PROJECT_SOURCE_DIR}/main_fep.cuh" "${PROJECT_SOURCE_DIR}/main_ti.cuh")
list(REMOVE_ITEM CURRENT_SOURCES "${PROJECT_SOURCE_DIR}/main.cu" "${PROJECT_SOURCE_DIR}/main_nopbc.cu"
  "${PROJECT_SOURCE_DIR}/main_fep.cu" "${PROJECT_SOURCE_DIR}/main_ti.cu")

source_group("Include" FILES ${CURRENT_HEADERS})
source_group("Source" FILES ${CURRENT_SOURCES})

cuda_add_library(sponge SHARED ${CURRENT_HEADERS} ${CURRENT_SOURCES})
cuda_add_executable(SPONGE main.cu)
target_link_libraries(SPONGE sponge ${CUDA_PATH}/lib64/libcufft.so)
cuda_add_executable(SPONGE_FEP main_fep.cu)
target_link_libraries(SPONGE_FEP sponge ${CUDA_PATH}/lib64/libcufft.so)
cuda_add_executable(SPONGE_TI main_ti.cu)
target_link_libraries(SPONGE_TI sponge ${CUDA_PATH}/lib64/libcufft.so)
cuda_add_executable(SPONGE_NOPBC main_nopbc.cu)
target_link_libraries(SPONGE_NOPBC sponge ${CUDA_PATH}/lib64/libcufft.so)
