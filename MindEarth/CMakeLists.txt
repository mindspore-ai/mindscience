cmake_minimum_required(VERSION 3.14.1)
project(MindEarth)

# set build options for project target
include(${CMAKE_SOURCE_DIR}/cmake/options.cmake)


find_package(Python3 COMPONENTS Interpreter Development)
if(Python3_FOUND)
    set(PYTHON_INCLUDE_DIRS "${Python3_INCLUDE_DIRS}")
    set(PYTHON_LIBRARIES "${Python3_LIBRARIES}")
else()
    find_python_package(py_inc py_lib)
    set(PYTHON_INCLUDE_DIRS "${py_inc}")
    set(PYTHON_LIBRARIES "${py_lib}")
endif()

message("PYTHON_INCLUDE_DIRS = ${PYTHON_INCLUDE_DIRS}")
message("PYTHON_LIBRARIES = ${PYTHON_LIBRARIES}")
include_directories(${PYTHON_INCLUDE_DIRS})

# packaging tool
include(${CMAKE_SOURCE_DIR}/cmake/package.cmake)
